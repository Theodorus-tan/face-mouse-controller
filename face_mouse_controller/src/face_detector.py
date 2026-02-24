"""
Face detection module using MediaPipe FaceLandmarker.
Handles model loading, face detection, and landmark extraction.
"""

import os
import urllib.request
import ssl
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import Config


@dataclass
class FaceLandmarks:
    """Data class for storing face landmark coordinates."""
    nose_tip: Tuple[int, int]
    forehead: Tuple[int, int]
    left_eye_inner: Tuple[int, int]
    right_eye_inner: Tuple[int, int]
    upper_lip_top: Tuple[int, int]
    upper_lip: Tuple[int, int]
    lower_lip: Tuple[int, int]
    lower_lip_bottom: Tuple[int, int]
    chin: Tuple[int, int]
    left_mouth_corner: Tuple[int, int]
    right_mouth_corner: Tuple[int, int]
    
    raw_coords: Dict[str, Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.raw_coords is None:
            self.raw_coords = {
                "nose_tip": self.nose_tip,
                "forehead": self.forehead,
                "left_eye_inner": self.left_eye_inner,
                "right_eye_inner": self.right_eye_inner,
                "upper_lip_top": self.upper_lip_top,
                "upper_lip": self.upper_lip,
                "lower_lip": self.lower_lip,
                "lower_lip_bottom": self.lower_lip_bottom,
                "chin": self.chin,
                "left_mouth_corner": self.left_mouth_corner,
                "right_mouth_corner": self.right_mouth_corner,
            }


@dataclass
class HeadPosition:
    """Data class for storing calculated head position."""
    center: Tuple[int, int]
    nose: Tuple[int, int]
    forehead: Tuple[int, int]
    tilt_x: int
    tilt_y: int


class FaceDetector:
    """
    Face detector using MediaPipe FaceLandmarker.
    Handles model download, initialization, and face detection.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the face detector.
        
        Args:
            config: Configuration object containing detector parameters
        """
        self.config = config
        self.detector: Optional[vision.FaceLandmarker] = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the face detector.
        Downloads model if necessary and creates the detector.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self._download_model():
            print("Failed to download model")
            return False
        
        self.detector = self._create_detector()
        if self.detector is None:
            return False
        
        self._initialized = True
        print("Face detector initialized successfully")
        return True
    
    def _download_model(self) -> bool:
        """
        Download the face landmarker model if not present.
        
        Returns:
            True if model available, False otherwise
        """
        model_path = self.config.model_path
        
        if os.path.exists(model_path):
            return True
        
        print("Downloading face landmarker model...")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            context = ssl._create_unverified_context()
            
            with urllib.request.urlopen(
                self.config.MODEL_URL, 
                context=context
            ) as response:
                with open(model_path, 'wb') as out_file:
                    out_file.write(response.read())
            
            print(f"Model downloaded: {model_path}")
            return True
            
        except Exception as e:
            print(f"Model download failed: {e}")
            return False
    
    def _create_detector(self) -> Optional[vision.FaceLandmarker]:
        """
        Create the FaceLandmarker detector instance.
        
        Returns:
            FaceLandmarker instance or None on failure
        """
        model_path = self.config.model_path
        
        if not os.path.exists(model_path):
            print("Model file not found")
            return None
        
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self.config.NUM_FACES,
                min_face_detection_confidence=self.config.MIN_FACE_DETECTION_CONFIDENCE,
                min_face_presence_confidence=self.config.MIN_FACE_PRESENCE_CONFIDENCE,
                min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            
            return vision.FaceLandmarker.create_from_options(options)
            
        except Exception as e:
            print(f"Detector creation failed: {e}")
            return None
    
    def detect(self, frame: np.ndarray, timestamp_ms: int) -> Optional[Any]:
        """
        Detect face landmarks in a frame.
        
        Args:
            frame: BGR image frame
            timestamp_ms: Frame timestamp in milliseconds
            
        Returns:
            Face landmarks or None if no face detected
        """
        if not self._initialized or self.detector is None:
            return None
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, 
                data=frame_rgb
            )
            
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            
            if result.face_landmarks and len(result.face_landmarks) > 0:
                return result.face_landmarks[0]
            
            return None
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return None
    
    def extract_landmarks(
        self, 
        landmarks: Any, 
        frame_shape: Tuple[int, int, int]
    ) -> Optional[FaceLandmarks]:
        """
        Extract landmark coordinates from detection result.
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: Frame dimensions (h, w, channels)
            
        Returns:
            FaceLandmarks object or None on failure
        """
        if landmarks is None:
            return None
        
        h, w = frame_shape[:2]
        coords = {}
        
        for idx, name in self.config.LANDMARK_INDICES.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                coords[name] = (int(landmark.x * w), int(landmark.y * h))
        
        required_keys = [
            "nose_tip", "forehead", "left_eye_inner", "right_eye_inner",
            "upper_lip_top", "upper_lip", "lower_lip", "lower_lip_bottom",
            "chin", "left_mouth_corner", "right_mouth_corner"
        ]
        
        if not all(key in coords for key in required_keys):
            return None
        
        return FaceLandmarks(
            nose_tip=coords["nose_tip"],
            forehead=coords["forehead"],
            left_eye_inner=coords["left_eye_inner"],
            right_eye_inner=coords["right_eye_inner"],
            upper_lip_top=coords["upper_lip_top"],
            upper_lip=coords["upper_lip"],
            lower_lip=coords["lower_lip"],
            lower_lip_bottom=coords["lower_lip_bottom"],
            chin=coords["chin"],
            left_mouth_corner=coords["left_mouth_corner"],
            right_mouth_corner=coords["right_mouth_corner"],
            raw_coords=coords
        )
    
    def calculate_head_position(
        self, 
        landmarks: FaceLandmarks
    ) -> HeadPosition:
        """
        Calculate head position from landmarks.
        
        Args:
            landmarks: Face landmarks object
            
        Returns:
            HeadPosition object
        """
        nose = landmarks.nose_tip
        forehead = landmarks.forehead
        
        tilt_x = forehead[0] - nose[0]
        tilt_y = forehead[1] - nose[1]
        
        return HeadPosition(
            center=nose,
            nose=nose,
            forehead=forehead,
            tilt_x=tilt_x,
            tilt_y=tilt_y
        )
    
    def detect_mouth_open(
        self, 
        landmarks: FaceLandmarks, 
        threshold: Optional[int] = None
    ) -> bool:
        """
        Detect if mouth is open.
        
        Args:
            landmarks: Face landmarks object
            threshold: Mouth open threshold (uses config if None)
            
        Returns:
            True if mouth is open
        """
        if threshold is None:
            threshold = self.config.MOUTH_OPEN_THRESHOLD
        
        mouth_top = landmarks.upper_lip_top
        mouth_bottom = landmarks.lower_lip_bottom
        
        mouth_open = abs(mouth_top[1] - mouth_bottom[1])
        
        return mouth_open > threshold
    
    def detect_blink(
        self, 
        landmarks: FaceLandmarks, 
        threshold: Optional[int] = None
    ) -> bool:
        """
        Detect if user is blinking.
        
        Args:
            landmarks: Face landmarks object
            threshold: Blink threshold (uses config if None)
            
        Returns:
            True if blink detected
        """
        if threshold is None:
            threshold = self.config.BLINK_THRESHOLD
        
        left_eye = landmarks.left_eye_inner
        right_eye = landmarks.right_eye_inner
        
        eye_distance = abs(left_eye[0] - right_eye[0])
        
        return eye_distance < threshold
    
    def close(self) -> None:
        """Close the detector and release resources."""
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        self._initialized = False
