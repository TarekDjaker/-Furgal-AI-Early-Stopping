"""Top‑level package for the Early Stopping extensions.

This package exposes the main modules and classes defined in the
enhanced early stopping library.  Users can import directly from
``early_stopping_enhanced`` to access the functionality.
"""

from .proximal_early_stopping import ProximalEarlyStopping, l1_proximal
from .component_early_stopping import ComponentEarlyStopping
from .fairness_early_stopping import (
    group_error_rates,
    demographic_parity_difference,
    FairnessEarlyStopping,
)
from .dp_early_stopping import DPSGDEarlyStopping

__all__ = [
    "ProximalEarlyStopping",
    "l1_proximal",
    "ComponentEarlyStopping",
    "group_error_rates",
    "demographic_parity_difference",
    "FairnessEarlyStopping",
    "DPSGDEarlyStopping",
]