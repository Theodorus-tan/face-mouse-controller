"""
Configuration module for Face-Controlled Mouse System.
Contains all configurable parameters and default settings.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict
import os


@dataclass
class Config:
    """Configuration class containing all system parameters."""
    
    MODEL_FILENAME: str = "face_landmarker.task"
    MODEL_URL: str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    CAMERA_WIDTH: int = 320
    CAMERA_HEIGHT: int = 240
    CAMERA_FPS: int = 30
    
    WINDOW_WIDTH: int = 800
    WINDOW_HEIGHT: int = 600
    WINDOW_NAME: str = "Face Controlled Mouse"
    
    SMOOTH_FACTOR: float = 0.5
    HISTORY_SIZE: int = 3
    DEAD_ZONE: int = 20
    JITTER_THRESHOLD: int = 5
    EFFECTIVE_RATIO: float = 0.2
    
    MOUTH_OPEN_THRESHOLD: int = 15
    BLINK_THRESHOLD: int = 20
    
    CLICK_COOLDOWN_FRAMES: int = 10
    MOUTH_COOLDOWN_FRAMES: int = 15
    
    MIN_FACE_DETECTION_CONFIDENCE: float = 0.7
    MIN_FACE_PRESENCE_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.7
    NUM_FACES: int = 1
    
    KALMAN_PROCESS_NOISE: float = 1e-5
    KALMAN_MEASUREMENT_NOISE: float = 1e-1
    KALMAN_INITIAL_ERROR: float = 1.0
    
    COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'face': (0, 255, 0),
        'nose': (255, 0, 0),
        'eye': (0, 255, 255),
        'cursor': (0, 0, 255),
        'text': (255, 255, 255),
    })
    
    LANDMARK_INDICES: Dict[int, str] = field(default_factory=lambda: {
        1: "nose_tip",
        10: "forehead",
        33: "left_eye_inner",
        263: "right_eye_inner",
        12: "upper_lip_top",
        13: "upper_lip",
        14: "lower_lip",
        15: "lower_lip_bottom",
        152: "chin",
        61: "left_mouth_corner",
        291: "right_mouth_corner",
    })
    
    @property
    def model_path(self) -> str:
        """Get the full path to the model file."""
        return os.path.join(os.path.dirname(__file__), "..", self.MODEL_FILENAME)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
