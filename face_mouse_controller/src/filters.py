"""
Filtering algorithms for smoothing mouse movements.
Includes Kalman Filter and Moving Average Filter implementations.
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class KalmanFilter:
    """
    Kalman Filter for state estimation and noise reduction.
    Implements a 2D state vector [position, velocity].
    """
    
    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-1,
        initial_estimate_error: float = 1.0
    ):
        """
        Initialize the Kalman Filter.
        
        Args:
            process_noise: Process noise covariance (Q matrix)
            measurement_noise: Measurement noise covariance (R matrix)
            initial_estimate_error: Initial estimation error (P matrix)
        """
        self.x = np.zeros(2)
        self.P = np.eye(2) * initial_estimate_error
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])
        self.I = np.eye(2)
    
    def predict(self) -> np.ndarray:
        """
        Perform the prediction step.
        
        Returns:
            Predicted state vector
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x.copy()
    
    def update(self, measurement: float) -> float:
        """
        Perform the update step with a new measurement.
        
        Args:
            measurement: Observed position value
            
        Returns:
            Estimated position after update
        """
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, (measurement - np.dot(self.H, self.x)))
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
        return self.x[0]
    
    def reset(self) -> None:
        """Reset the filter to initial state."""
        self.x = np.zeros(2)
        self.P = np.eye(2)


class MovingAverageFilter:
    """
    Moving Average Filter for temporal smoothing.
    Uses a fixed-size window to average recent values.
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize the Moving Average Filter.
        
        Args:
            window_size: Number of samples to average
        """
        self.window_size = window_size
        self.history_x: deque = deque(maxlen=window_size)
        self.history_y: deque = deque(maxlen=window_size)
    
    def filter(self, x: int, y: int) -> Tuple[int, int]:
        """
        Apply moving average filter to coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Filtered (x, y) coordinates
        """
        self.history_x.append(x)
        self.history_y.append(y)
        
        filtered_x = int(sum(self.history_x) / len(self.history_x))
        filtered_y = int(sum(self.history_y) / len(self.history_y))
        
        return filtered_x, filtered_y
    
    def reset(self) -> None:
        """Clear the filter history."""
        self.history_x.clear()
        self.history_y.clear()
    
    @property
    def is_ready(self) -> bool:
        """Check if the filter has enough samples."""
        return len(self.history_x) >= self.window_size


class ExponentialSmoother:
    """
    Exponential smoothing filter for responsive yet smooth output.
    Blends current input with previous output.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize the Exponential Smoother.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother
        """
        self.alpha = alpha
        self.prev_x: Optional[int] = None
        self.prev_y: Optional[int] = None
    
    def smooth(self, x: int, y: int) -> Tuple[int, int]:
        """
        Apply exponential smoothing to coordinates.
        
        Args:
            x: Current X coordinate
            y: Current Y coordinate
            
        Returns:
            Smoothed (x, y) coordinates
        """
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y
        
        smooth_x = int(self.prev_x * self.alpha + x * (1 - self.alpha))
        smooth_y = int(self.prev_y * self.alpha + y * (1 - self.alpha))
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y
    
    def reset(self) -> None:
        """Reset the smoother state."""
        self.prev_x = None
        self.prev_y = None


class CompositeFilter:
    """
    Composite filter combining multiple smoothing techniques.
    Applies exponential smoothing followed by moving average.
    """
    
    def __init__(
        self,
        smooth_factor: float = 0.5,
        history_size: int = 3,
        use_kalman: bool = False,
        kalman_process_noise: float = 1e-5,
        kalman_measurement_noise: float = 1e-1
    ):
        """
        Initialize the composite filter.
        
        Args:
            smooth_factor: Exponential smoothing factor
            history_size: Moving average window size
            use_kalman: Whether to use Kalman filtering
            kalman_process_noise: Kalman process noise
            kalman_measurement_noise: Kalman measurement noise
        """
        self.exponential = ExponentialSmoother(alpha=smooth_factor)
        self.moving_avg = MovingAverageFilter(window_size=history_size)
        self.use_kalman = use_kalman
        
        if use_kalman:
            self.kalman_x = KalmanFilter(
                process_noise=kalman_process_noise,
                measurement_noise=kalman_measurement_noise
            )
            self.kalman_y = KalmanFilter(
                process_noise=kalman_process_noise,
                measurement_noise=kalman_measurement_noise
            )
    
    def filter(self, x: int, y: int) -> Tuple[int, int]:
        """
        Apply composite filtering to coordinates.
        
        Args:
            x: Raw X coordinate
            y: Raw Y coordinate
            
        Returns:
            Filtered (x, y) coordinates
        """
        if self.use_kalman:
            x = int(self.kalman_x.update(x))
            y = int(self.kalman_y.update(y))
        
        smooth_x, smooth_y = self.exponential.smooth(x, y)
        return self.moving_avg.filter(smooth_x, smooth_y)
    
    def reset(self) -> None:
        """Reset all filters."""
        self.exponential.reset()
        self.moving_avg.reset()
        if self.use_kalman:
            self.kalman_x.reset()
            self.kalman_y.reset()
