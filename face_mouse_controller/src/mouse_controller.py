"""
Mouse controller module.
Handles screen coordinate mapping and mouse operations.
"""

from typing import Optional, Tuple
import pyautogui

from .config import Config
from .filters import CompositeFilter
from .face_detector import HeadPosition, FaceLandmarks


class MouseController:
    """
    Controls mouse movement and clicks based on face detection.
    Handles coordinate mapping, smoothing, and click detection.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the mouse controller.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.screen_w, self.screen_h = pyautogui.size()
        
        pyautogui.FAILSAFE = False
        
        self.filter = CompositeFilter(
            smooth_factor=config.SMOOTH_FACTOR,
            history_size=config.HISTORY_SIZE,
            use_kalman=False
        )
        
        self.prev_x: int = 0
        self.prev_y: int = 0
        
        self.click_anchor: Optional[Tuple[int, int]] = None
        self.mouth_was_open: bool = False
        
        self._click_cooldown: int = 0
        self._mouth_cooldown: int = 0
    
    def map_to_screen(
        self, 
        head_pos: HeadPosition, 
        frame_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Map head position to screen coordinates.
        
        Args:
            head_pos: Head position from face detector
            frame_shape: Frame dimensions
            
        Returns:
            Screen coordinates (x, y) or None
        """
        if head_pos is None:
            return None
        
        h, w = frame_shape[:2]
        center = head_pos.center
        
        effective_ratio = self.config.EFFECTIVE_RATIO
        
        effective_w = w * effective_ratio
        effective_h = h * effective_ratio
        effective_x = (w - effective_w) // 2
        effective_y = (h - effective_h) // 2
        
        rel_x = (center[0] - effective_x) / effective_w
        rel_y = (center[1] - effective_y) / effective_h
        
        rel_x = max(0, min(1, rel_x))
        rel_y = max(0, min(1, rel_y))
        
        target_x = int(rel_x * self.screen_w)
        target_y = int(rel_y * self.screen_h)
        
        target_x, target_y = self._apply_dead_zone(target_x, target_y)
        
        target_x, target_y = self._apply_jitter_control(target_x, target_y)
        
        screen_x, screen_y = self.filter.filter(target_x, target_y)
        
        screen_x = max(0, min(self.screen_w - 1, screen_x))
        screen_y = max(0, min(self.screen_h - 1, screen_y))
        
        self.prev_x, self.prev_y = screen_x, screen_y
        
        return (screen_x, screen_y)
    
    def _apply_dead_zone(
        self, 
        target_x: int, 
        target_y: int
    ) -> Tuple[int, int]:
        """
        Apply dead zone filtering to reduce small movements.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            
        Returns:
            Filtered coordinates
        """
        dead_zone = self.config.DEAD_ZONE
        dx = target_x - self.prev_x
        dy = target_y - self.prev_y
        
        if abs(dx) < dead_zone and abs(dy) < dead_zone and self.click_anchor is None:
            return self.prev_x, self.prev_y
        
        return target_x, target_y
    
    def _apply_jitter_control(
        self, 
        target_x: int, 
        target_y: int
    ) -> Tuple[int, int]:
        """
        Apply jitter control during click operations.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            
        Returns:
            Stabilized coordinates
        """
        if self.click_anchor is not None:
            dx = target_x - self.click_anchor[0]
            dy = target_y - self.click_anchor[1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            
            if distance < self.config.JITTER_THRESHOLD:
                return self.click_anchor
        
        return target_x, target_y
    
    def update_mouth_state(
        self, 
        is_mouth_open: bool
    ) -> Optional[str]:
        """
        Update mouth state and detect click gesture.
        
        Args:
            is_mouth_open: Current mouth open state
            
        Returns:
            'left_click' if click triggered, None otherwise
        """
        if is_mouth_open and not self.mouth_was_open:
            self.click_anchor = (self.prev_x, self.prev_y)
            action = self._trigger_click('left')
            self.mouth_was_open = True
            return action
        
        elif not is_mouth_open:
            self.click_anchor = None
            self.mouth_was_open = False
        
        return None
    
    def _trigger_click(self, button: str) -> Optional[str]:
        """
        Trigger a mouse click with cooldown.
        
        Args:
            button: 'left' or 'right'
            
        Returns:
            Click action name or None if on cooldown
        """
        if self._click_cooldown > 0:
            return None
        
        try:
            if button == 'left':
                pyautogui.click(button='left')
                self._click_cooldown = self.config.CLICK_COOLDOWN_FRAMES
                return 'left_click'
            elif button == 'right':
                pyautogui.click(button='right')
                self._click_cooldown = self.config.CLICK_COOLDOWN_FRAMES
                return 'right_click'
        except Exception as e:
            print(f"Click error: {e}")
        
        return None
    
    def trigger_blink_click(self) -> Optional[str]:
        """
        Trigger right click on blink detection.
        
        Returns:
            'right_click' if triggered, None otherwise
        """
        return self._trigger_click('right')
    
    def move_to(self, x: int, y: int) -> bool:
        """
        Move mouse to specified coordinates.
        
        Args:
            x: Screen X coordinate
            y: Screen Y coordinate
            
        Returns:
            True if successful
        """
        try:
            pyautogui.moveTo(x, y)
            return True
        except Exception as e:
            print(f"Mouse move error: {e}")
            return False
    
    def update_cooldowns(self) -> None:
        """Decrement cooldown counters."""
        if self._click_cooldown > 0:
            self._click_cooldown -= 1
        if self._mouth_cooldown > 0:
            self._mouth_cooldown -= 1
    
    def reset(self) -> None:
        """Reset controller state."""
        self.filter.reset()
        self.prev_x, self.prev_y = 0, 0
        self.click_anchor = None
        self.mouth_was_open = False
        self._click_cooldown = 0
        self._mouth_cooldown = 0
