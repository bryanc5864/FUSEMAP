"""
FUSEMAP Trainer

Main training loop with:
- Multi-dataset training with balanced sampling
- Comprehensive logging
- Per-epoch validation
- Early stopping
- Checkpoint management
- Mixed precision training
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn


from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from .config import ExperimentConfig, ConfigurationType, DATASET_CATALOG
from .datasets import (
    MultiDataset, ActivityNormalizer, collate_multi_dataset,
    get_validation_loader, SingleDataset
)
from .samplers import TemperatureBalancedSampler, GlobalIndexSampler, ExtremeAwareSampler, BalancedActivitySampler
from .models import (
    MultiSpeciesCADENCE, create_multi_species_model, compute_masked_loss
)
from .metrics import MetricsTracker, compute_all_metrics, DREAMYeastMetrics


class Trainer:
    """
    Main trainer class for FUSEMAP experiments.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        device: str = "cuda",
        resume_from: Optional[str] = None,
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resume_from = resume_from

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.normalizer = ActivityNormalizer()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.training.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('-inf')
        self.patience_counter = 0

        # Metrics tracking
        self.metrics_tracker = None

    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / "training.log"

        # Create logger
        self.logger = logging.getLogger(self.config.name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(f"Experiment: {self.config.name}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def setup_data(self):
        """Setup datasets and data loaders."""
        self.logger.info("Setting up datasets...")

        # Create training dataset
        self.train_dataset = MultiDataset(
            dataset_names=self.config.datasets,
            split="train",
            target_length=self.config.target_sequence_length,
            normalizer=self.normalizer,
        )

        # Get dataset sizes for balanced sampling
        dataset_sizes = self.train_dataset.get_dataset_sizes()
        self.logger.info(f"Dataset sizes: {dataset_sizes}")

        # Create sampler - priority: balanced > extreme > global
        use_balanced_sampling = getattr(self.config.training, 'use_balanced_sampling', False)
        use_extreme_sampling = getattr(self.config.training, 'use_extreme_sampling', False)

        # Collect activities for activity-aware samplers
        # For multi-output datasets, average across outputs to get single activity value
        all_activities = []
        for ds_name, ds in self.train_dataset.datasets.items():
            acts = ds.activities
            if acts.ndim > 1:
                # Multi-output: average across outputs for sampling purposes
                acts = acts.mean(axis=1)
            all_activities.append(acts)
        all_activities = np.concatenate(all_activities, axis=0)

        if use_balanced_sampling:
            # Balanced sampling: equal samples from each activity bin
            n_bins = getattr(self.config.training, 'balanced_sampling_bins', 10)
            self.logger.info(f"Using balanced activity sampling with {n_bins} bins...")
            self.sampler = BalancedActivitySampler(
                activities=all_activities,
                dataset_sizes=dataset_sizes,
                n_bins=n_bins,
                temperature=self.config.training.sampling_temperature,
                samples_per_epoch=self.config.training.samples_per_epoch,
                seed=self.config.seed,
            )
        elif use_extreme_sampling:
            # Extreme-aware sampling (deprecated)
            self.logger.info("Using extreme-aware sampling...")
            sampling_alpha = getattr(self.config.training, 'sampling_extreme_alpha', 0.5)
            self.sampler = ExtremeAwareSampler(
                activities=all_activities,
                dataset_sizes=dataset_sizes,
                extreme_alpha=sampling_alpha,
                extreme_beta=self.config.training.extreme_beta,
                temperature=self.config.training.sampling_temperature,
                samples_per_epoch=self.config.training.samples_per_epoch,
                seed=self.config.seed,
            )
            weight_stats = self.sampler.get_weight_statistics()
            for name, stats in weight_stats.items():
                self.logger.info(f"  {name}: weights min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
        else:
            self.logger.info("Using standard global index sampling...")
            self.sampler = GlobalIndexSampler(
                dataset_sizes=dataset_sizes,
                temperature=self.config.training.sampling_temperature,
                samples_per_epoch=self.config.training.samples_per_epoch,
                seed=self.config.seed,
            )

        # Create train loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_multi_dataset,
        )

        # Create validation loaders for each dataset
        # Use index mappings from train_dataset for consistency
        index_mappings = {
            "species": self.train_dataset.species_to_idx,
            "kingdom": self.train_dataset.kingdom_to_idx,
            "celltype": self.train_dataset.celltype_to_idx,
        }

        self.val_loaders = {}
        self.val_types = {}
        self.train_eval_loaders = {}  # For evaluating on train set

        for dataset_name in self.config.datasets:
            try:
                loader, val_type = get_validation_loader(
                    dataset_name,
                    target_length=self.config.target_sequence_length,
                    batch_size=self.config.training.batch_size,
                    normalizer=self.normalizer,
                    index_mappings=index_mappings,
                )
                self.val_loaders[dataset_name] = loader
                self.val_types[dataset_name] = val_type
                self.logger.info(f"  {dataset_name}: {val_type} set loaded")
            except Exception as e:
                self.logger.warning(f"  {dataset_name}: Could not load validation data: {e}")

        # Create train eval loaders (sequential, for evaluation only)
        # We'll subsample to speed up evaluation
        for dataset_name in self.config.datasets:
            try:
                info = DATASET_CATALOG[dataset_name]
                train_eval_dataset = SingleDataset(
                    info, "train",
                    self.config.target_sequence_length,
                    self.normalizer,
                    index_mappings=index_mappings,
                )
                self.train_eval_loaders[dataset_name] = DataLoader(
                    train_eval_dataset,
                    batch_size=self.config.training.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    collate_fn=collate_multi_dataset,
                )
            except Exception as e:
                self.logger.warning(f"  {dataset_name}: Could not load train eval data: {e}")

        # Create test and calibration loaders for datasets with held-out splits
        self.test_loaders = {}
        self.calibration_loaders = {}

        # Datasets that support test/calibration splits
        # - encode4_*: Human MPRA with test/calibration splits
        # - dream_yeast: Yeast with special test set + 1% calibration from train
        # - deepstarr: Drosophila with chromosome-based test split
        # - jores_*: Plant with standard test splits
        datasets_with_test = [
            "encode4_k562", "encode4_hepg2", "encode4_wtc11",
            "dream_yeast", "deepstarr",
            "jores_arabidopsis", "jores_maize", "jores_sorghum"
        ]
        datasets_with_calibration = [
            "encode4_k562", "encode4_hepg2", "encode4_wtc11",
            "dream_yeast"
        ]

        for dataset_name in self.config.datasets:
            # Skip datasets without test splits
            if dataset_name not in datasets_with_test:
                continue

            info = DATASET_CATALOG[dataset_name]

            # Test loader
            try:
                test_dataset = SingleDataset(
                    info, "test",
                    self.config.target_sequence_length,
                    self.normalizer,
                    index_mappings=index_mappings,
                )
                self.test_loaders[dataset_name] = DataLoader(
                    test_dataset,
                    batch_size=self.config.training.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    collate_fn=collate_multi_dataset,
                )
                self.logger.info(f"  {dataset_name}: test set loaded ({len(test_dataset)} samples)")
            except Exception as e:
                self.logger.warning(f"  {dataset_name}: Could not load test data: {e}")

            # Calibration loader (only for datasets that support it)
            if dataset_name in datasets_with_calibration:
                try:
                    calib_dataset = SingleDataset(
                        info, "calibration",
                        self.config.target_sequence_length,
                        self.normalizer,
                        index_mappings=index_mappings,
                    )
                    self.calibration_loaders[dataset_name] = DataLoader(
                        calib_dataset,
                        batch_size=self.config.training.batch_size,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        collate_fn=collate_multi_dataset,
                    )
                    self.logger.info(f"  {dataset_name}: calibration set loaded ({len(calib_dataset)} samples)")
                except Exception as e:
                    self.logger.warning(f"  {dataset_name}: Could not load calibration data: {e}")

        # Setup metrics tracker
        output_names = {
            name: DATASET_CATALOG[name].output_names
            for name in self.config.datasets
            if name in DATASET_CATALOG
        }
        self.metrics_tracker = MetricsTracker(
            dataset_names=self.config.datasets,
            output_names_per_dataset=output_names,
        )

        # Save normalizer
        self.normalizer.save(str(self.output_dir / "normalizer.json"))
        self.logger.info("Data setup complete")

    def setup_model(self):
        """Setup model, optimizer, and scheduler."""
        self.logger.info("Setting up model...")

        # Create model with correct embedding sizes from train_dataset
        self.model = create_multi_species_model(
            config=self.config.model,
            dataset_names=self.config.datasets,
            n_species=len(self.train_dataset.species_to_idx),
            n_kingdoms=len(self.train_dataset.kingdom_to_idx),
            n_celltypes=len(self.train_dataset.celltype_to_idx),
        )
        self.model = self.model.to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {n_params:,} ({n_trainable:,} trainable)")

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Setup scheduler
        if self.config.training.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.training.max_epochs,
                eta_min=self.config.training.min_lr,
            )
        elif self.config.training.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=self.config.training.min_lr,
            )
        elif self.config.training.scheduler == "onecycle":
            # OneCycleLR requires total_steps (train_loader already set up)
            steps_per_epoch = len(self.train_loader)
            total_steps = steps_per_epoch * self.config.training.max_epochs
            div_factor = self.config.training.onecycle_div_factor
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.training.onecycle_pct_start,
                div_factor=div_factor,  # initial_lr = max_lr / div_factor
            )
            initial_lr = self.config.training.learning_rate / div_factor
            self.logger.info(f"OneCycleLR: {total_steps} total steps, "
                           f"initial_lr={initial_lr:.6f}, max_lr={self.config.training.learning_rate}")
        else:
            self.scheduler = None

        # Resume if specified
        if self.resume_from:
            self.load_checkpoint(self.resume_from)

        self.logger.info("Model setup complete")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.sampler.set_epoch(self.current_epoch)

        epoch_losses = []  # NLL loss (can be negative with uncertainty)
        epoch_mse_losses = []  # MSE loss (always positive, for monitoring)
        epoch_grad_norms = []
        per_head_losses = {name: [] for name in self.model.heads.keys()}

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )

        accumulation_steps = self.config.training.gradient_accumulation_steps
        accumulated_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            sequence = batch['sequence'].to(self.device)
            activity = batch['activity'].to(self.device)
            species_idx = batch['species_idx'].to(self.device)
            kingdom_idx = batch['kingdom_idx'].to(self.device)
            celltype_idx = batch['celltype_idx'].to(self.device)
            original_length = batch['original_length'].to(self.device)
            dataset_names = batch['dataset_names']

            # Forward pass with optional AMP
            with autocast(enabled=self.config.training.use_amp):
                outputs = self.model(
                    sequence=sequence,
                    species_idx=species_idx,
                    kingdom_idx=kingdom_idx,
                    celltype_idx=celltype_idx,
                    original_length=original_length,
                    dataset_names=dataset_names,
                )

                # Compute loss (with extreme value weighting)
                loss, head_losses = compute_masked_loss(
                    outputs=outputs,
                    targets=activity,
                    dataset_names=dataset_names,
                    dataset_to_heads=self.model.dataset_to_heads,
                    use_uncertainty=self.config.model.use_uncertainty,
                    use_extreme_weights=self.config.training.use_extreme_weights,
                    extreme_alpha=self.config.training.extreme_alpha,
                    extreme_beta=self.config.training.extreme_beta,
                )

                # Also compute MSE loss for monitoring (without weighting)
                mse_loss, _ = compute_masked_loss(
                    outputs=outputs,
                    targets=activity,
                    dataset_names=dataset_names,
                    dataset_to_heads=self.model.dataset_to_heads,
                    use_uncertainty=False,  # Pure MSE
                    use_extreme_weights=False,  # No weighting for monitoring
                )

                loss = loss / accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Step OneCycleLR scheduler after each batch (not epoch)
                if self.scheduler and isinstance(
                    self.scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    self.scheduler.step()

                self.optimizer.zero_grad()

                # Record metrics
                epoch_losses.append(accumulated_loss)
                mse_val = mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss
                epoch_mse_losses.append(mse_val)
                epoch_grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

                for name, h_loss in head_losses.items():
                    per_head_losses[name].append(h_loss.item())

                accumulated_loss = 0.0
                self.global_step += 1

                # Update progress bar with both losses
                pbar.set_postfix({
                    'nll': f"{epoch_losses[-1]:.4f}",
                    'mse': f"{epoch_mse_losses[-1]:.4f}",
                    'grad': f"{epoch_grad_norms[-1]:.2f}",
                })

                # Log periodically
                if self.global_step % self.config.training.log_every_n_steps == 0:
                    self.logger.info(
                        f"Step {self.global_step}: NLL={epoch_losses[-1]:.4f}, "
                        f"MSE={epoch_mse_losses[-1]:.4f}, "
                        f"grad={epoch_grad_norms[-1]:.2f}, "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

        # Epoch summary
        metrics = {
            'train_loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'train_mse': np.mean(epoch_mse_losses) if epoch_mse_losses else 0.0,
            'train_grad_norm': np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0,
        }

        for name, losses in per_head_losses.items():
            if losses:
                metrics[f'train_loss_{name}'] = np.mean(losses)

        return metrics

    @torch.no_grad()
    def evaluate_loader(
        self,
        loaders: Dict[str, DataLoader],
        split_name: str = "val",
        max_batches: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict], Dict[str, float]]:
        """
        Evaluate on a set of data loaders.

        Returns:
            all_metrics: Per-dataset, per-output metrics (Pearson, Spearman, R2, etc.)
            aggregate_metrics: Aggregate loss metrics (NLL, MSE) across all data
        """
        self.model.eval()

        all_metrics = {}

        # Track losses across all batches
        total_nll = 0.0
        total_mse = 0.0
        total_samples = 0

        for dataset_name, loader in loaders.items():
            predictions = []
            targets = []
            log_vars = []  # For NLL computation
            weights = []

            for batch_idx, batch in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break

                sequence = batch['sequence'].to(self.device)
                activity = batch['activity'].to(self.device)
                species_idx = batch['species_idx'].to(self.device)
                kingdom_idx = batch['kingdom_idx'].to(self.device)
                celltype_idx = batch['celltype_idx'].to(self.device)
                original_length = batch['original_length'].to(self.device)
                ds_names = batch['dataset_names']

                # Forward pass
                outputs = self.model(
                    sequence=sequence,
                    species_idx=species_idx,
                    kingdom_idx=kingdom_idx,
                    celltype_idx=celltype_idx,
                    original_length=original_length,
                    dataset_names=ds_names,
                )

                # Compute losses for this batch (no weighting for evaluation)
                nll_loss, _ = compute_masked_loss(
                    outputs=outputs,
                    targets=activity,
                    dataset_names=ds_names,
                    dataset_to_heads=self.model.dataset_to_heads,
                    use_uncertainty=self.config.model.use_uncertainty,
                    use_extreme_weights=False,  # No weighting for evaluation
                )
                mse_loss, _ = compute_masked_loss(
                    outputs=outputs,
                    targets=activity,
                    dataset_names=ds_names,
                    dataset_to_heads=self.model.dataset_to_heads,
                    use_uncertainty=False,
                    use_extreme_weights=False,  # No weighting for evaluation
                )

                batch_size = sequence.shape[0]
                total_nll += nll_loss.item() * batch_size
                total_mse += mse_loss.item() * batch_size
                total_samples += batch_size

                # Extract predictions for this dataset's heads
                heads = self.model.dataset_to_heads.get(dataset_name, [])
                batch_preds = []
                batch_logvars = []

                for i, head_name in enumerate(heads):
                    if head_name in outputs:
                        pred_data = outputs[head_name]
                        head_preds = pred_data['mean'].cpu()
                        batch_preds.append(head_preds)
                        if 'log_var' in pred_data:
                            batch_logvars.append(pred_data['log_var'].cpu())

                if batch_preds:
                    batch_preds = torch.stack(batch_preds, dim=-1)
                    predictions.append(batch_preds)
                    targets.append(activity[:, :len(heads)].cpu())
                    if batch_logvars:
                        log_vars.append(torch.stack(batch_logvars, dim=-1))

                if 'weight' in batch:
                    weights.append(batch['weight'])

            if not predictions:
                continue

            predictions = torch.cat(predictions, dim=0).numpy()
            targets_np = torch.cat(targets, dim=0).numpy()

            # Handle single output
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            if targets_np.ndim == 1:
                targets_np = targets_np.reshape(-1, 1)

            # Compute metrics per output
            info = DATASET_CATALOG.get(dataset_name)
            output_names = info.output_names if info else ["output"]

            dataset_metrics = {}
            for i, output_name in enumerate(output_names):
                if i >= predictions.shape[1]:
                    continue

                # Inverse transform predictions for fair comparison
                pred_orig = self.normalizer.inverse_transform(
                    dataset_name,
                    predictions[:, i],
                )
                target_orig = self.normalizer.inverse_transform(
                    dataset_name,
                    targets_np[:, i],
                )

                metrics = compute_all_metrics(target_orig, pred_orig)
                dataset_metrics[output_name] = metrics.to_dict()

            # Special handling for DREAM yeast
            if dataset_name == "dream_yeast" and weights:
                dream_metrics = DREAMYeastMetrics()
                weights_arr = torch.cat(weights, dim=0).numpy()
                dream_results = dream_metrics.compute_dream_score(
                    targets_np[:, 0],
                    predictions[:, 0],
                    weights=weights_arr,
                )
                dataset_metrics["dream_score"] = dream_results

            all_metrics[dataset_name] = dataset_metrics

        # Compute aggregate metrics
        aggregate_metrics = {
            'nll': total_nll / total_samples if total_samples > 0 else 0.0,
            'mse': total_mse / total_samples if total_samples > 0 else 0.0,
        }

        return all_metrics, aggregate_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, Dict]:
        """Run validation on all datasets (backward compatible)."""
        all_metrics, _ = self.evaluate_loader(self.val_loaders, "val")
        return all_metrics

    @torch.no_grad()
    def _evaluate_held_out_sets(
        self,
        loaders: Dict[str, torch.utils.data.DataLoader],
        split_name: str = "test"
    ) -> Dict[str, Dict]:
        """Evaluate on held-out test or calibration sets.

        Args:
            loaders: Dict mapping dataset_name -> DataLoader
            split_name: Name of the split (for logging)

        Returns:
            Dictionary with metrics per dataset
        """
        self.model.eval()
        all_metrics = {}

        for dataset_name, loader in loaders.items():
            self.logger.info(f"  Evaluating {dataset_name} ({split_name})...")

            predictions = []
            targets = []
            log_vars = []

            for batch in loader:
                sequence = batch['sequence'].to(self.device)
                activity = batch['activity'].to(self.device)
                species_idx = batch['species_idx'].to(self.device)
                kingdom_idx = batch['kingdom_idx'].to(self.device)
                celltype_idx = batch['celltype_idx'].to(self.device)
                original_length = batch['original_length'].to(self.device)
                ds_names = batch['dataset_names']

                # Forward pass
                outputs = self.model(
                    sequence=sequence,
                    species_idx=species_idx,
                    kingdom_idx=kingdom_idx,
                    celltype_idx=celltype_idx,
                    original_length=original_length,
                    dataset_names=ds_names,
                )

                # Extract predictions for this dataset's heads
                heads = self.model.dataset_to_heads.get(dataset_name, [])
                batch_preds = []
                batch_logvars = []

                for head_name in heads:
                    if head_name in outputs:
                        pred_data = outputs[head_name]
                        batch_preds.append(pred_data['mean'].cpu())
                        if 'logvar' in pred_data:
                            batch_logvars.append(pred_data['logvar'].cpu())

                if batch_preds:
                    batch_preds = torch.stack(batch_preds, dim=-1)
                    predictions.append(batch_preds)
                    targets.append(activity[:, :len(heads)].cpu())
                    if batch_logvars:
                        log_vars.append(torch.stack(batch_logvars, dim=-1))

            if not predictions:
                self.logger.warning(f"    No predictions for {dataset_name}")
                continue

            predictions = torch.cat(predictions, dim=0).numpy()
            targets_np = torch.cat(targets, dim=0).numpy()

            # Handle single output
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            if targets_np.ndim == 1:
                targets_np = targets_np.reshape(-1, 1)

            # Compute metrics per output
            info = DATASET_CATALOG.get(dataset_name)
            output_names = info.output_names if info else [f"output_{i}" for i in range(predictions.shape[1])]

            dataset_metrics = {}
            for i, output_name in enumerate(output_names):
                if i >= predictions.shape[1]:
                    break

                preds = predictions[:, i]
                targs = targets_np[:, i]

                # Remove NaNs
                valid_mask = ~np.isnan(targs) & ~np.isnan(preds)
                if valid_mask.sum() < 2:
                    continue

                preds = preds[valid_mask]
                targs = targs[valid_mask]

                # Compute metrics
                from scipy.stats import pearsonr, spearmanr
                pearson_r, _ = pearsonr(preds, targs)
                spearman_r, _ = spearmanr(preds, targs)
                mse = np.mean((preds - targs) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(preds - targs))

                # R2
                ss_res = np.sum((targs - preds) ** 2)
                ss_tot = np.sum((targs - np.mean(targs)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                dataset_metrics[output_name] = {
                    'pearson': pearson_r,
                    'spearman': spearman_r,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'n_samples': len(preds),
                }

                self.logger.info(
                    f"    {output_name}: r={pearson_r:.4f}, rho={spearman_r:.4f}, "
                    f"R2={r2:.4f}, RMSE={rmse:.4f} (n={len(preds)})"
                )

            all_metrics[dataset_name] = dataset_metrics

        return all_metrics

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_metric': self.best_val_metric,
            'config': asdict(self.config),
        }

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_val_metric = checkpoint.get('best_val_metric', float('-inf'))

        self.logger.info(f"Resumed from epoch {self.current_epoch}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {self.config.config_type.value}")

        # Setup
        self.setup_data()
        self.setup_model()

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        # Training loop
        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate on both train and val sets
            if epoch % self.config.training.val_every_n_epochs == 0:
                # Evaluate on train set (limit batches for speed)
                max_train_batches = 50  # Limit to ~50 batches for train eval
                train_eval_metrics, train_agg = self.evaluate_loader(
                    self.train_eval_loaders, "train", max_batches=max_train_batches
                )

                # Evaluate on val set (full evaluation)
                val_metrics, val_agg = self.evaluate_loader(self.val_loaders, "val")

                # Helper to extract aggregate metrics from per-dataset metrics
                def extract_aggregate(metrics_dict):
                    pearsons, spearmans, r2s, rmses, maes = [], [], [], [], []
                    for ds_metrics in metrics_dict.values():
                        for output_metrics in ds_metrics.values():
                            if isinstance(output_metrics, dict) and 'pearson' in output_metrics:
                                pearsons.append(output_metrics['pearson']['value'])
                                spearmans.append(output_metrics['spearman']['value'])
                                r2s.append(output_metrics['r2']['value'])
                                rmses.append(output_metrics['rmse']['value'])
                                maes.append(output_metrics['mae']['value'])
                    return {
                        'pearson': np.mean(pearsons) if pearsons else 0.0,
                        'spearman': np.mean(spearmans) if spearmans else 0.0,
                        'r2': np.mean(r2s) if r2s else 0.0,
                        'rmse': np.mean(rmses) if rmses else 0.0,
                        'mae': np.mean(maes) if maes else 0.0,
                    }

                train_corr = extract_aggregate(train_eval_metrics)
                val_corr = extract_aggregate(val_metrics)

                avg_val_pearson = val_corr['pearson']

                # Check if best
                is_best = avg_val_pearson > self.best_val_metric

                if is_best:
                    self.best_val_metric = avg_val_pearson
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Log comprehensive epoch summary with SAME stats for train and val
                self.logger.info("=" * 80)
                self.logger.info(f"EPOCH {epoch} SUMMARY {'(BEST)' if is_best else ''}")
                self.logger.info("=" * 80)
                self.logger.info(
                    f"[TRAIN] NLL: {train_agg['nll']:.4f} | MSE: {train_agg['mse']:.4f} | "
                    f"r: {train_corr['pearson']:.4f} | rho: {train_corr['spearman']:.4f} | "
                    f"R2: {train_corr['r2']:.4f} | RMSE: {train_corr['rmse']:.4f}"
                )
                self.logger.info(
                    f"[VAL]   NLL: {val_agg['nll']:.4f} | MSE: {val_agg['mse']:.4f} | "
                    f"r: {val_corr['pearson']:.4f} | rho: {val_corr['spearman']:.4f} | "
                    f"R2: {val_corr['r2']:.4f} | RMSE: {val_corr['rmse']:.4f}"
                )
                self.logger.info("-" * 80)

                # Log per-dataset metrics for validation
                for ds_name, ds_metrics in val_metrics.items():
                    self.logger.info(f"  [VAL] {ds_name}:")
                    for output_name, metrics in ds_metrics.items():
                        if isinstance(metrics, dict) and 'pearson' in metrics:
                            self.logger.info(
                                f"    {output_name}: "
                                f"r={metrics['pearson']['value']:.4f} | "
                                f"rho={metrics['spearman']['value']:.4f} | "
                                f"R2={metrics['r2']['value']:.4f} | "
                                f"RMSE={metrics['rmse']['value']:.4f} | "
                                f"MAE={metrics['mae']['value']:.4f}"
                            )
                self.logger.info("=" * 80)

                # Save checkpoint
                self.save_checkpoint(
                    self.output_dir / f"checkpoint_epoch{epoch}.pt",
                    is_best=is_best,
                )

                # Early stopping check
                if self.patience_counter >= self.config.training.patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(patience={self.config.training.patience})"
                    )
                    break

            # Update scheduler (skip OneCycleLR - it's stepped per batch in train_epoch)
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_pearson)
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        # Final evaluation
        self.logger.info("Training complete. Running final evaluation...")
        final_metrics = self.validate()

        # Evaluate on test and calibration sets (for human MPRA datasets)
        if hasattr(self, 'test_loaders') and self.test_loaders:
            self.logger.info("\n" + "="*60)
            self.logger.info("Evaluating on TEST sets...")
            self.logger.info("="*60)
            test_metrics = self._evaluate_held_out_sets(self.test_loaders, "test")
            final_metrics['test'] = test_metrics

        if hasattr(self, 'calibration_loaders') and self.calibration_loaders:
            self.logger.info("\n" + "="*60)
            self.logger.info("Evaluating on CALIBRATION sets...")
            self.logger.info("="*60)
            calib_metrics = self._evaluate_held_out_sets(self.calibration_loaders, "calibration")
            final_metrics['calibration'] = calib_metrics

        # Save final results
        results_path = self.output_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)

        self.logger.info(f"Results saved to {results_path}")

        return final_metrics


class MultiPhaseTrainer(Trainer):
    """
    Trainer for multi-phase training (Config 5: Universal Foundation Model).
    """

    def train(self):
        """Multi-phase training loop."""
        self.logger.info("Starting multi-phase training...")

        phases = self.config.training_phases or []

        if not phases:
            # Fall back to single-phase training
            return super().train()

        # Setup
        self.setup_data()
        self.setup_model()

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        for phase_idx, phase in enumerate(phases):
            phase_name = phase.get('name', f'phase_{phase_idx}')
            phase_epochs = phase.get('epochs', 50)
            freeze_backbone = phase.get('freeze_backbone', False)
            phase_lr = phase.get('lr', self.config.training.learning_rate)

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"PHASE {phase_idx + 1}: {phase_name}")
            self.logger.info(f"  Epochs: {phase_epochs}")
            self.logger.info(f"  Freeze backbone: {freeze_backbone}")
            self.logger.info(f"  Learning rate: {phase_lr}")
            self.logger.info(f"{'='*60}\n")

            # Apply phase settings
            if freeze_backbone:
                self.model.freeze_backbone()
            else:
                self.model.unfreeze_backbone()

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = phase_lr

            # Reset scheduler for this phase
            if self.config.training.scheduler == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=phase_epochs,
                    eta_min=self.config.training.min_lr,
                )

            # Reset patience for this phase
            self.patience_counter = 0
            phase_start_epoch = self.current_epoch

            # Train phase
            for epoch in range(phase_epochs):
                self.current_epoch = phase_start_epoch + epoch
                epoch_start = time.time()

                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Compute average validation metric
                val_pearsons = []
                for ds_metrics in val_metrics.values():
                    for output_metrics in ds_metrics.values():
                        if isinstance(output_metrics, dict) and 'pearson' in output_metrics:
                            val_pearsons.append(output_metrics['pearson']['value'])

                avg_val_pearson = np.mean(val_pearsons) if val_pearsons else 0.0

                is_best = avg_val_pearson > self.best_val_metric
                if is_best:
                    self.best_val_metric = avg_val_pearson
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                self.logger.info(
                    f"[{phase_name}] Epoch {epoch}: "
                    f"loss={train_metrics['train_loss']:.4f}, "
                    f"val_r={avg_val_pearson:.4f} "
                    f"{'(BEST)' if is_best else ''}"
                )

                # Save checkpoint
                self.save_checkpoint(
                    self.output_dir / f"checkpoint_{phase_name}_epoch{epoch}.pt",
                    is_best=is_best,
                )

                # Early stopping within phase
                if self.patience_counter >= self.config.training.patience // 2:
                    self.logger.info(f"Early stopping phase {phase_name}")
                    break

                if self.scheduler:
                    self.scheduler.step()

                epoch_time = time.time() - epoch_start
                self.logger.info(f"Epoch completed in {epoch_time:.1f}s")

        # Final evaluation
        self.logger.info("Multi-phase training complete. Running final evaluation...")
        final_metrics = self.validate()

        results_path = self.output_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)

        return final_metrics
