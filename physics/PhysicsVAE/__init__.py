"""
PhysicsVAE: Physics-Conditioned Variational Autoencoder for Sequence Generation

A VAE that generates DNA sequences conditioned on target biophysical properties,
enabling physics-constrained sequence design and optimization.
"""

from .models.physics_vae import PhysicsVAE, PhysicsEncoder, SequenceEncoder, SequenceDecoder
from .models.losses import VAELoss, PhysicsConsistencyLoss

__version__ = "0.1.0"
