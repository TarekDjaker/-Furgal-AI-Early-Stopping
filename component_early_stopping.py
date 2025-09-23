"""
Component‑wise early stopping for iterative algorithms.

This module provides a utility class that monitors the loss of individual
components in multi-component learning algorithms and decides when to stop
training specific components based on their convergence behavior. The idea is
to exploit heterogeneous convergence rates across different components, allowing
faster components to stop early while slower components continue training.

This can be applied to ensemble methods, multi-task learning, or any algorithm
where you can track performance per component.
"""

from __future__ import annotations

from typing import Dict, List, Set
import numpy as np


class ComponentEarlyStopping:
    """Monitor and stop training of individual components based on their convergence.

    Parameters
    ----------
    n_components : int
        Number of components to monitor.
    patience : int, optional
        Number of iterations to wait for improvement before stopping a component.
        Defaults to 10.
    min_delta : float, optional
        Minimum change in loss to qualify as an improvement. Defaults to 1e-4.
    verbose : bool, optional
        If True, prints a message whenever a component is stopped.
    """

    def __init__(self, n_components: int, patience: int = 10, min_delta: float = 1e-4, verbose: bool = False) -> None:
        self.n_components = n_components
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        # Track best loss and patience counter for each component
        self.best_losses: Dict[int, float] = {}
        self.patience_counters: Dict[int, int] = {}
        self.stopped_components: Dict[int, int] = {}  # component_id -> iteration_stopped
        self.loss_histories: Dict[int, List[float]] = {i: [] for i in range(n_components)}

    def should_stop_component(self, component_id: int, loss: float, iteration: int) -> bool:
        """Check if a component should stop training.
        
        Parameters
        ----------
        component_id : int
            ID of the component (0 to n_components-1).
        loss : float
            Current loss value for this component.
        iteration : int
            Current iteration number.
            
        Returns
        -------
        bool
            True if the component should stop training.
        """
        if component_id in self.stopped_components:
            return True  # Already stopped
            
        # Record loss history
        self.loss_histories[component_id].append(loss)
        
        # Initialize tracking for this component if first time
        if component_id not in self.best_losses:
            self.best_losses[component_id] = loss
            self.patience_counters[component_id] = 0
            return False
            
        # Check if loss improved
        if loss < self.best_losses[component_id] - self.min_delta:
            self.best_losses[component_id] = loss
            self.patience_counters[component_id] = 0
            return False
        else:
            self.patience_counters[component_id] += 1
            
        # Check if patience exceeded
        if self.patience_counters[component_id] >= self.patience:
            self.stopped_components[component_id] = iteration
            if self.verbose:
                print(f"ComponentEarlyStopping: stopped component {component_id} at iteration {iteration}")
            return True
            
        return False
        
    def get_active_components(self, iteration: int) -> Set[int]:
        """Get the set of components that are still active at given iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
            
        Returns
        -------
        Set[int]
            Set of component IDs that are still active.
        """
        active = set(range(self.n_components))
        for component_id, stop_iter in self.stopped_components.items():
            if stop_iter <= iteration:
                active.discard(component_id)
        return active
        
    def get_computational_savings(self, total_iterations: int) -> float:
        """Calculate computational savings as percentage.
        
        Parameters
        ----------
        total_iterations : int
            Total number of iterations that would have been run.
            
        Returns
        -------
        float
            Percentage of computational savings (0-100).
        """
        if not self.stopped_components:
            return 0.0
            
        total_computations = self.n_components * total_iterations
        saved_computations = 0
        
        for component_id, stop_iter in self.stopped_components.items():
            saved_computations += max(0, total_iterations - stop_iter)
            
        return 100.0 * saved_computations / total_computations
        
    def reset(self) -> None:
        """Reset all tracking state."""
        self.best_losses.clear()
        self.patience_counters.clear()
        self.stopped_components.clear()
        self.loss_histories = {i: [] for i in range(self.n_components)}
        
    def summary(self) -> Dict[str, any]:
        """Return a summary of the stopping behavior.
        
        Returns
        -------
        Dict[str, any]
            Summary statistics including stopped components and savings.
        """
        return {
            'n_components': self.n_components,
            'stopped_components': dict(self.stopped_components),
            'n_stopped': len(self.stopped_components),
            'active_components': list(self.get_active_components(float('inf'))),
            'patience': self.patience,
            'min_delta': self.min_delta
        }


__all__ = ["ComponentEarlyStopping"]