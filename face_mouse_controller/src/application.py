"""
Main application module for Face-Controlled Mouse System.
Integrates all components and handles the main loop.
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .face_detector import FaceDetector, FaceLandmarks, HeadPosition
from .mouse_controller import MouseController


@dataclass
class FrameData:
    """Container for processed frame data."""
    frame: np.ndarray
    landmarks: Optional[FaceLandmarks]
    head_position: Optional[HeadPosition]
    screen_pos: Optional[Tuple[int, int]]
    is_mouth_open: bool
    is_blinking: bool
    fps: float


class Visualizer:
    """Handles all visualization and drawing operations."""
    
    def __init__(self, config: Config):
        """Initialize the visualizer."""
        self.config = config
        self.colors = config.COLORS
    
    def draw_overlay(
        self, 
        frame: np.ndarray,
        landmarks: Optional[FaceLandmarks],
        head_pos: Optional[HeadPosition],
        screen_pos: Optional[Tuple[int, int]],
        fps: float
    ) -> np.ndarray:
        """
        Draw visualization overlay on frame.
        
        Args:
            frame: Input frame
            landmarks: Face landmarks
            head_pos: Head position
            screen_pos: Screen coordinates
            fps: Current FPS
            
        Returns:
            Frame with overlay
        """
        output = frame.copy()
        
        if landmarks is not None:
            output = self._draw_landmarks(output, landmarks)
        
        if head_pos is not None:
            output = self._draw_head_position(output, head_pos)
        
        if screen_pos is not None:
            output = self._draw_cursor(output, screen_pos)
        
        output = self._draw_info_panel(output, screen_pos, fps)
        
        return output
    
    def _draw_landmarks(
        self, 
        frame: np.ndarray, 
        landmarks: FaceLandmarks
    ) -> np.ndarray:
        """Draw face landmarks on frame."""
        for name, point in landmarks.raw_coords.items():
            color = self.colors['nose'] if name == "nose_tip" else self.colors['face']
            cv2.circle(frame, point, 3, color, -1)
        
        return frame
    
    def _draw_head_position(
        self, 
        frame: np.ndarray, 
        head_pos: HeadPosition
    ) -> np.ndarray:
        """Draw head position indicator."""
        center = head_pos.center
        cv2.circle(frame, center, 8, (0, 255, 255), -1)
        
        cv2.line(
            frame, 
            head_pos.forehead, 
            head_pos.nose, 
            (255, 255, 0), 
            2
        )
        
        return frame
    
    def _draw_cursor(
        self, 
        frame: np.ndarray, 
        screen_pos: Tuple[int, int]
    ) -> np.ndarray:
        """Draw virtual cursor on frame."""
        h, w = frame.shape[:2]
        
        display_x = int((screen_pos[0] / self.config.screen_w) * w)
        display_y = int((screen_pos[1] / self.config.screen_h) * h)
        
        color = self.colors['cursor']
        cv2.circle(frame, (display_x, display_y), 10, color, 2)
        cv2.line(frame, (display_x - 15, display_y), (display_x + 15, display_y), color, 2)
        cv2.line(frame, (display_x, display_y - 15), (display_x, display_y + 15), color, 2)
        
        return frame
    
    def _draw_info_panel(
        self, 
        frame: np.ndarray, 
        screen_pos: Optional[Tuple[int, int]],
        fps: float
    ) -> np.ndarray:
        """Draw information panel on frame."""
        h, w = frame.shape[:2]
        panel_x, panel_y = 10, 30
        
        cv2.putText(
            frame, "Face Controlled Mouse",
            (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            self.colors['text'], 2
        )
        panel_y += 30
        
        if screen_pos:
            coord_text = f"Screen: ({screen_pos[0]}, {screen_pos[1]})"
            cv2.putText(
                frame, coord_text,
                (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                self.colors['text'], 1
            )
            panel_y += 25
        
        status_texts = [
            "Head Move = Mouse Move",
            "Open Mouth = Left Click",
            "Blink = Right Click",
            "'c' Calibrate | 'q' Quit"
        ]
        
        for text in status_texts:
            cv2.putText(
                frame, text,
                (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                self.colors['text'], 1
            )
            panel_y += 25
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text, (w - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        
        return frame


class FaceMouseApplication:
    """
    Main application class for Face-Controlled Mouse System.
    Coordinates all components and runs the main loop.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the application.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self.face_detector = FaceDetector(self.config)
        self.mouse_controller = MouseController(self.config)
        self.visualizer = Visualizer(self.config)
        
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        
        self._frame_count = 0
        self._fps = 0.0
        self._last_time = time.time()
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if all components initialized successfully
        """
        print("=" * 50)
        print("Face-Controlled Mouse System")
        print("=" * 50)
        print("Initializing...")
        
        if not self.face_detector.initialize():
            print("Failed to initialize face detector")
            return False
        
        print("All components initialized successfully")
        return True
    
    def _setup_camera(self, camera_id: int = 0) -> bool:
        """
        Set up the camera capture.
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            True if camera opened successfully
        """
        print(f"Opening camera {camera_id}...")
        
        self._cap = cv2.VideoCapture(camera_id)
        
        if not self._cap.isOpened():
            print("Failed to open camera")
            print("Troubleshooting:")
            print("  1. Ensure camera is not used by another application")
            print("  2. Grant camera access in System Preferences")
            print("  3. Try a different camera_id")
            return False
        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        print("Camera opened successfully")
        return True
    
    def run(self, camera_id: int = 0) -> None:
        """
        Run the main application loop.
        
        Args:
            camera_id: Camera device ID
        """
        if not self.initialize():
            return
        
        if not self._setup_camera(camera_id):
            return
        
        self._print_instructions()
        
        cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.config.WINDOW_NAME,
            self.config.WINDOW_WIDTH,
            self.config.WINDOW_HEIGHT
        )
        
        self._running = True
        timestamp_ms = 0
        
        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                frame_data = self._process_frame(frame, timestamp_ms)
                
                if frame_data.screen_pos:
                    self.mouse_controller.move_to(
                        frame_data.screen_pos[0],
                        frame_data.screen_pos[1]
                    )
                
                output_frame = self.visualizer.draw_overlay(
                    frame_data.frame,
                    frame_data.landmarks,
                    frame_data.head_position,
                    frame_data.screen_pos,
                    frame_data.fps
                )
                
                cv2.imshow(self.config.WINDOW_NAME, output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self._calibrate()
                elif key == ord('s'):
                    self._save_screenshot(output_frame)
                
                self.mouse_controller.update_cooldowns()
                timestamp_ms += 33
                self._update_fps()
                
        finally:
            self._cleanup()
    
    def _process_frame(
        self, 
        frame: np.ndarray, 
        timestamp_ms: int
    ) -> FrameData:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            timestamp_ms: Frame timestamp
            
        Returns:
            Processed frame data
        """
        landmarks = None
        head_position = None
        screen_pos = None
        is_mouth_open = False
        is_blinking = False
        
        raw_landmarks = self.face_detector.detect(frame, timestamp_ms)
        
        if raw_landmarks is not None:
            landmarks = self.face_detector.extract_landmarks(
                raw_landmarks, frame.shape
            )
            
            if landmarks is not None:
                head_position = self.face_detector.calculate_head_position(landmarks)
                
                screen_pos = self.mouse_controller.map_to_screen(
                    head_position, frame.shape
                )
                
                is_mouth_open = self.face_detector.detect_mouth_open(landmarks)
                action = self.mouse_controller.update_mouth_state(is_mouth_open)
                
                is_blinking = self.face_detector.detect_blink(landmarks)
                if is_blinking:
                    self.mouse_controller.trigger_blink_click()
        
        return FrameData(
            frame=frame,
            landmarks=landmarks,
            head_position=head_position,
            screen_pos=screen_pos,
            is_mouth_open=is_mouth_open,
            is_blinking=is_blinking,
            fps=self._fps
        )
    
    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self._frame_count += 1
        current_time = time.time()
        time_diff = current_time - self._last_time
        
        if time_diff >= 1.0:
            self._fps = self._frame_count / time_diff
            self._frame_count = 0
            self._last_time = current_time
    
    def _calibrate(self) -> None:
        """Perform calibration."""
        print("Calibration started...")
        print("Look at the four corners of your screen")
        self.mouse_controller.reset()
    
    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save a screenshot."""
        filename = f"face_mouse_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    
    def _print_instructions(self) -> None:
        """Print usage instructions."""
        print("\n" + "=" * 50)
        print("Instructions:")
        print("=" * 50)
        print("  - Face the camera, keep your face visible")
        print("  - Move your head to control the mouse")
        print("  - Open mouth to left-click")
        print("  - Blink to right-click")
        print("  - Press 'c' to calibrate")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("=" * 50 + "\n")
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        print("\nShutting down...")
        
        if self._cap is not None:
            self._cap.release()
        
        cv2.destroyAllWindows()
        self.face_detector.close()
        
        print("Goodbye!")
    
    def stop(self) -> None:
        """Stop the application."""
        self._running = False
