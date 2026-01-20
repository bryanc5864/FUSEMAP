"""
FUSEMAP Balanced Sampling Strategies

Implements various sampling strategies for handling dataset imbalance:
- Temperature-scaled sampling
- Gradient normalization balancing
- Per-dataset epoch balancing
"""

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from typing import Dict, List, Optional, Iterator, Tuple
import math


class TemperatureBalancedSampler(Sampler):
    """
    Sample datasets with temperature scaling.

    τ = 1.0: proportional to size (dominated by large datasets)
    τ = 0.0: uniform across datasets (equal representation)
    τ = 0.5: balanced middle ground (recommended)

    Args:
        dataset_sizes: Dict mapping dataset name to size
        temperature: Temperature parameter (0.0 to 1.0)
        samples_per_epoch: Total samples per epoch (None = geometric mean)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset_sizes: Dict[str, int],
        temperature: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset_sizes = dataset_sizes
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0

        # Compute sampling probabilities
        sizes = np.array(list(dataset_sizes.values()))
        self.dataset_names = list(dataset_sizes.keys())

        # Temperature scaling: p_i ∝ n_i^τ
        weights = sizes ** temperature
        self.probs = weights / weights.sum()

        # Total samples per epoch
        if samples_per_epoch is None:
            # Use geometric mean of dataset sizes
            self.samples_per_epoch = int(np.exp(np.mean(np.log(sizes))))
        else:
            self.samples_per_epoch = samples_per_epoch

        # Compute cumulative offsets for global indexing
        self.cumsum = np.cumsum([0] + list(dataset_sizes.values()))
        self.dataset_to_offset = {
            name: self.cumsum[i]
            for i, name in enumerate(self.dataset_names)
        }

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """
        Yield (dataset_name, local_index) tuples.
        """
        rng = np.random.RandomState(self.seed + self.epoch)

        for _ in range(self.samples_per_epoch):
            # Pick a dataset based on temperature-scaled probabilities
            dataset_idx = rng.choice(len(self.dataset_names), p=self.probs)
            dataset_name = self.dataset_names[dataset_idx]

            # Pick a random sample from that dataset
            sample_idx = rng.randint(self.dataset_sizes[dataset_name])

            yield (dataset_name, sample_idx)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def get_effective_samples_per_dataset(self) -> Dict[str, int]:
        """Get expected samples per dataset per epoch."""
        return {
            name: int(self.samples_per_epoch * prob)
            for name, prob in zip(self.dataset_names, self.probs)
        }


class GlobalIndexSampler(Sampler):
    """
    Sampler that returns global indices across multiple datasets.

    Useful when datasets are concatenated and you need global indexing.
    """

    def __init__(
        self,
        dataset_sizes: Dict[str, int],
        temperature: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.base_sampler = TemperatureBalancedSampler(
            dataset_sizes, temperature, samples_per_epoch, seed
        )

        # Build offset map
        self.offsets = {}
        current_offset = 0
        for name, size in dataset_sizes.items():
            self.offsets[name] = current_offset
            current_offset += size

    def set_epoch(self, epoch: int):
        self.base_sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[int]:
        for dataset_name, local_idx in self.base_sampler:
            yield self.offsets[dataset_name] + local_idx

    def __len__(self) -> int:
        return len(self.base_sampler)


class StratifiedMultiDatasetSampler(Sampler):
    """
    Stratified sampling that ensures each batch has samples from multiple datasets.

    Useful for multi-task learning where gradients from different tasks
    should be balanced within each batch.
    """

    def __init__(
        self,
        dataset_sizes: Dict[str, int],
        batch_size: int,
        min_samples_per_dataset: int = 1,
        temperature: float = 0.5,
        seed: int = 42,
    ):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.min_samples_per_dataset = min_samples_per_dataset
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0

        self.dataset_names = list(dataset_sizes.keys())
        n_datasets = len(self.dataset_names)

        # Compute per-batch allocation
        # At least min_samples_per_dataset from each, rest by temperature
        base_per_dataset = min_samples_per_dataset
        remaining = batch_size - base_per_dataset * n_datasets

        if remaining < 0:
            raise ValueError(
                f"batch_size {batch_size} too small for "
                f"{n_datasets} datasets with min {min_samples_per_dataset} each"
            )

        # Temperature-scaled allocation of remaining
        sizes = np.array(list(dataset_sizes.values()))
        weights = sizes ** temperature
        probs = weights / weights.sum()

        self.samples_per_batch = {
            name: base_per_dataset + int(remaining * prob)
            for name, prob in zip(self.dataset_names, probs)
        }

        # Compute total samples
        total_size = sum(dataset_sizes.values())
        self.n_batches = total_size // batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[Tuple[str, int]]]:
        """Yield batches as lists of (dataset_name, local_idx) tuples."""
        rng = np.random.RandomState(self.seed + self.epoch)

        # Shuffle indices within each dataset
        shuffled_indices = {
            name: rng.permutation(size)
            for name, size in self.dataset_sizes.items()
        }
        current_pos = {name: 0 for name in self.dataset_names}

        for _ in range(self.n_batches):
            batch = []

            for name in self.dataset_names:
                n_samples = self.samples_per_batch[name]

                for _ in range(n_samples):
                    # Wrap around if exhausted
                    if current_pos[name] >= len(shuffled_indices[name]):
                        shuffled_indices[name] = rng.permutation(
                            self.dataset_sizes[name]
                        )
                        current_pos[name] = 0

                    idx = shuffled_indices[name][current_pos[name]]
                    batch.append((name, idx))
                    current_pos[name] += 1

            # Shuffle batch for better gradient mixing
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


class GradNormBalancer:
    """
    Balance gradients across tasks by adjusting loss weights dynamically.

    Based on GradNorm paper: https://arxiv.org/abs/1711.02257
    """

    def __init__(
        self,
        n_tasks: int,
        alpha: float = 1.5,
        update_rate: float = 0.1,
    ):
        """
        Args:
            n_tasks: Number of tasks
            alpha: Restoring force strength (higher = more aggressive balancing)
            update_rate: How fast to update weights
        """
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.update_rate = update_rate

        self.weights = torch.ones(n_tasks)
        self.initial_losses = None
        self.loss_history = [[] for _ in range(n_tasks)]

    def update(
        self,
        losses: List[torch.Tensor],
        shared_params: List[torch.nn.Parameter],
    ) -> torch.Tensor:
        """
        Update weights based on gradient norms.

        Args:
            losses: List of per-task losses
            shared_params: List of shared parameters to compute gradients on

        Returns:
            Updated weight tensor
        """
        # Store initial losses
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in losses]

        # Track losses
        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss.item())

        # Compute gradient norms per task
        grad_norms = []
        for i, loss in enumerate(losses):
            # Compute gradients w.r.t. shared params
            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            grad_norm = sum(
                g.norm() for g in grads if g is not None
            )
            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms)

        # Compute relative task difficulty
        relative_losses = torch.tensor([
            l.item() / l0 for l, l0 in zip(losses, self.initial_losses)
        ])
        avg_relative = relative_losses.mean()

        # Target gradient norms
        avg_grad_norm = grad_norms.mean()
        targets = avg_grad_norm * (relative_losses / avg_relative) ** self.alpha

        # Update weights (slow update for stability)
        weight_updates = targets / (grad_norms + 1e-8)
        self.weights = self.weights * (1 - self.update_rate) + \
                       weight_updates * self.update_rate

        # Normalize so weights sum to n_tasks
        self.weights = self.weights * self.n_tasks / self.weights.sum()

        return self.weights

    def get_weighted_loss(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of losses."""
        return sum(w * l for w, l in zip(self.weights, losses))


class DatasetWeightScheduler:
    """
    Schedule dataset weights throughout training.

    Can implement curriculum learning or other scheduling strategies.
    """

    def __init__(
        self,
        dataset_names: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        schedule: str = "constant",  # constant, linear, cosine
        warmup_epochs: int = 0,
    ):
        self.dataset_names = dataset_names
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs

        if initial_weights is None:
            self.initial_weights = {name: 1.0 for name in dataset_names}
        else:
            self.initial_weights = initial_weights

        self.current_weights = self.initial_weights.copy()

    def step(self, epoch: int, max_epochs: int):
        """Update weights for current epoch."""
        if self.schedule == "constant":
            return

        progress = epoch / max_epochs

        if epoch < self.warmup_epochs:
            # During warmup, use initial weights
            self.current_weights = self.initial_weights.copy()
        elif self.schedule == "linear":
            # Linear increase toward uniform
            for name in self.dataset_names:
                target = 1.0
                self.current_weights[name] = (
                    self.initial_weights[name] * (1 - progress) +
                    target * progress
                )
        elif self.schedule == "cosine":
            # Cosine schedule toward uniform
            for name in self.dataset_names:
                target = 1.0
                factor = 0.5 * (1 + math.cos(math.pi * progress))
                self.current_weights[name] = (
                    target + (self.initial_weights[name] - target) * factor
                )

    def get_weights(self) -> Dict[str, float]:
        return self.current_weights.copy()


class ReplayBuffer:
    """
    Experience replay buffer for difficult samples.

    Stores samples that had high loss and replays them more frequently.
    """

    def __init__(
        self,
        capacity: int = 10000,
        replay_fraction: float = 0.1,
    ):
        self.capacity = capacity
        self.replay_fraction = replay_fraction
        self.buffer: List[Tuple[str, int, float]] = []

    def add(
        self,
        dataset_name: str,
        sample_idx: int,
        loss: float,
    ):
        """Add sample with its loss to buffer."""
        self.buffer.append((dataset_name, sample_idx, loss))

        # Keep top-k by loss
        if len(self.buffer) > self.capacity:
            self.buffer.sort(key=lambda x: x[2], reverse=True)
            self.buffer = self.buffer[:self.capacity]

    def sample(self, n: int) -> List[Tuple[str, int]]:
        """Sample n items from buffer."""
        if not self.buffer:
            return []

        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [(self.buffer[i][0], self.buffer[i][1]) for i in indices]

    def get_replay_samples(self, batch_size: int) -> List[Tuple[str, int]]:
        """Get samples to replay based on replay_fraction."""
        n_replay = int(batch_size * self.replay_fraction)
        return self.sample(n_replay)


class BalancedActivitySampler(Sampler):
    """
    Sampler that creates uniform distribution across activity bins.

    Instead of natural distribution (many middle, few extremes), this sampler
    ensures equal representation from all activity ranges. Extremes get repeated
    more, middle values less - creating a balanced training signal.

    Args:
        activities: Activity values for all samples [N]
        dataset_sizes: Dict mapping dataset name to size
        n_bins: Number of activity bins (default 10 for deciles)
        temperature: Temperature for dataset balancing
        samples_per_epoch: Total samples per epoch
        seed: Random seed
    """

    def __init__(
        self,
        activities: np.ndarray,
        dataset_sizes: Dict[str, int],
        n_bins: int = 10,
        temperature: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.n_bins = n_bins
        self.seed = seed
        self.epoch = 0

        self.dataset_sizes = dataset_sizes
        self.dataset_names = list(dataset_sizes.keys())

        # Compute offsets for global indexing
        self.offsets = {}
        current_offset = 0
        for name, size in dataset_sizes.items():
            self.offsets[name] = current_offset
            current_offset += size
        self.total_size = current_offset

        # Bin samples by activity within each dataset
        self._create_bins(activities)

        # Samples per epoch
        if samples_per_epoch is None:
            self.samples_per_epoch = self.total_size
        else:
            self.samples_per_epoch = samples_per_epoch

        # Temperature-based dataset probabilities
        sizes = np.array(list(dataset_sizes.values()))
        weights = sizes ** temperature
        self.dataset_probs = weights / weights.sum()

    def _create_bins(self, activities: np.ndarray):
        """Create activity bins for each dataset using equal-width VALUE ranges.

        This ensures extremes (which have fewer samples) get repeated more often
        to achieve balanced coverage across the entire activity range.
        """
        if activities.ndim > 1:
            activities = np.nanmean(activities, axis=-1)

        self.bins_per_dataset = {}
        self.bin_edges_per_dataset = {}

        offset = 0
        for name, size in self.dataset_sizes.items():
            ds_activities = activities[offset:offset + size]
            valid_activities = ds_activities[~np.isnan(ds_activities)]

            # Use equal-width bins based on VALUE RANGE (not quantiles!)
            # This means extreme bins will have fewer samples -> more repetition
            min_val, max_val = valid_activities.min(), valid_activities.max()
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            bin_edges[0] = -np.inf  # Catch any outliers
            bin_edges[-1] = np.inf

            # Assign samples to bins
            bin_indices = np.digitize(ds_activities, bin_edges[1:-1])  # 0 to n_bins-1

            # Store indices for each bin
            bins = [[] for _ in range(self.n_bins)]
            for i, bin_idx in enumerate(bin_indices):
                bins[min(bin_idx, self.n_bins - 1)].append(i)

            # Convert to arrays
            self.bins_per_dataset[name] = [np.array(b) if len(b) > 0 else np.array([0]) for b in bins]
            self.bin_edges_per_dataset[name] = bin_edges

            offset += size

        # Log bin statistics
        for name in self.dataset_names:
            bin_sizes = [len(b) for b in self.bins_per_dataset[name]]
            total = sum(bin_sizes)
            print(f"  {name} bin sizes: {bin_sizes}")
            print(f"  {name} bin %: {[f'{100*s/total:.1f}%' for s in bin_sizes]}")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _precompute_indices(self, rng: np.random.RandomState) -> np.ndarray:
        """Precompute all indices with sqrt-balanced activity sampling.

        Uses square-root scaling: samples_from_bin ~ sqrt(bin_size)
        This balances between:
        - Pure proportional (middle dominates, extremes ignored)
        - Pure equal per bin (extremes massively over-represented)

        Result: extremes get ~3-5x more representation, not 20-30x
        """
        dataset_counts = rng.multinomial(self.samples_per_epoch, self.dataset_probs)

        all_indices = []

        for ds_name, ds_count in zip(self.dataset_names, dataset_counts):
            if ds_count == 0:
                continue

            bins = self.bins_per_dataset[ds_name]

            # Compute sqrt-scaled weights for each bin
            bin_sizes = np.array([max(len(b), 1) for b in bins])
            # sqrt scaling: balance between proportional and equal
            bin_weights = np.sqrt(bin_sizes)
            bin_probs = bin_weights / bin_weights.sum()

            # Allocate samples to bins based on sqrt-scaled probabilities
            bin_sample_counts = rng.multinomial(ds_count, bin_probs)

            ds_indices = []
            for bin_idx, (bin_samples, n_samples) in enumerate(zip(bins, bin_sample_counts)):
                if len(bin_samples) == 0 or n_samples == 0:
                    continue

                # Sample with replacement from this bin
                sampled = rng.choice(bin_samples, size=n_samples, replace=True)
                ds_indices.append(sampled)

            if ds_indices:
                ds_indices = np.concatenate(ds_indices)
                global_indices = self.offsets[ds_name] + ds_indices
                all_indices.append(global_indices)

        all_indices = np.concatenate(all_indices)
        rng.shuffle(all_indices)
        return all_indices

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = self._precompute_indices(rng)
        for idx in indices:
            yield int(idx)

    def __len__(self) -> int:
        return self.samples_per_epoch


class ExtremeAwareSampler(Sampler):
    """
    Sampler that oversamples extreme values (tails of the activity distribution).

    This helps the model learn to predict extreme values better by showing
    them more frequently during training.

    Args:
        activities: Array of activity values for all samples [N] or [N, outputs]
        dataset_sizes: Dict mapping dataset name to size (for multi-dataset)
        extreme_alpha: How much to weight extremes (0 = uniform, higher = more extremes)
        extreme_beta: Power for z-score (2.0 = quadratic emphasis)
        temperature: Temperature for dataset balancing
        samples_per_epoch: Total samples per epoch
        seed: Random seed
    """

    def __init__(
        self,
        activities: np.ndarray,
        dataset_sizes: Dict[str, int],
        extreme_alpha: float = 1.0,
        extreme_beta: float = 2.0,
        temperature: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.extreme_alpha = extreme_alpha
        self.extreme_beta = extreme_beta
        self.seed = seed
        self.epoch = 0

        self.dataset_sizes = dataset_sizes
        self.dataset_names = list(dataset_sizes.keys())

        # Compute cumulative offsets for global indexing
        self.offsets = {}
        current_offset = 0
        for name, size in dataset_sizes.items():
            self.offsets[name] = current_offset
            current_offset += size
        self.total_size = current_offset

        # Compute sample weights based on extreme values
        self._compute_weights(activities)

        # Total samples per epoch (default: total dataset size)
        if samples_per_epoch is None:
            self.samples_per_epoch = self.total_size
        else:
            self.samples_per_epoch = samples_per_epoch

        # Temperature-based dataset probabilities
        sizes = np.array(list(dataset_sizes.values()))
        weights = sizes ** temperature
        self.dataset_probs = weights / weights.sum()

    def _compute_weights(self, activities: np.ndarray):
        """Compute sample weights that emphasize extreme values."""
        # Handle multi-output by taking mean
        if activities.ndim > 1:
            activities = np.nanmean(activities, axis=-1)

        # Compute z-scores per dataset
        weights = np.ones(len(activities), dtype=np.float32)

        offset = 0
        for name, size in self.dataset_sizes.items():
            ds_activities = activities[offset:offset + size]

            # Z-score within this dataset
            mean = np.nanmean(ds_activities)
            std = np.nanstd(ds_activities) + 1e-8
            z_scores = np.abs((ds_activities - mean) / std)
            z_scores = np.clip(z_scores, 0, 4.0)  # Cap at 4 std

            # Weight formula: 1 + alpha * |z|^beta
            ds_weights = 1.0 + self.extreme_alpha * (z_scores ** self.extreme_beta)

            # Handle NaN
            ds_weights = np.nan_to_num(ds_weights, nan=1.0)

            weights[offset:offset + size] = ds_weights
            offset += size

        # Normalize within each dataset for proper sampling
        self.sample_weights = weights

        # Compute per-dataset normalized probabilities
        self.per_dataset_probs = {}
        offset = 0
        for name, size in self.dataset_sizes.items():
            ds_weights = self.sample_weights[offset:offset + size]
            ds_probs = ds_weights / ds_weights.sum()
            self.per_dataset_probs[name] = ds_probs
            offset += size

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch

    def _precompute_indices(self, rng: np.random.RandomState) -> np.ndarray:
        """Precompute all indices for the epoch (much faster than per-sample)."""
        # Determine how many samples per dataset
        dataset_counts = rng.multinomial(
            self.samples_per_epoch, self.dataset_probs
        )

        all_indices = []
        for i, (name, count) in enumerate(zip(self.dataset_names, dataset_counts)):
            if count == 0:
                continue
            # Sample from this dataset using precomputed probabilities
            probs = self.per_dataset_probs[name]
            # Use vectorized sampling - sample all at once
            local_indices = rng.choice(len(probs), size=count, p=probs, replace=True)
            global_indices = self.offsets[name] + local_indices
            all_indices.append(global_indices)

        # Concatenate and shuffle
        all_indices = np.concatenate(all_indices)
        rng.shuffle(all_indices)
        return all_indices

    def __iter__(self) -> Iterator[int]:
        """Yield global indices with extreme-aware sampling."""
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = self._precompute_indices(rng)
        for idx in indices:
            yield int(idx)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about sampling weights per dataset."""
        stats = {}
        offset = 0
        for name, size in self.dataset_sizes.items():
            ds_weights = self.sample_weights[offset:offset + size]
            stats[name] = {
                "min": float(ds_weights.min()),
                "max": float(ds_weights.max()),
                "mean": float(ds_weights.mean()),
                "std": float(ds_weights.std()),
            }
            offset += size
        return stats
