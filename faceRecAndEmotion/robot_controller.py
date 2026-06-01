"""
Robot Controller for Physical Movements and Reactions
Simulates robot responses based on detected emotions
"""

import time
import threading
import random
from config import ROBOT_REACTIONS

class RobotController:
    def __init__(self):
        self.current_emotion = "neutral"
        self.current_person = None
        self.tail_wag_state = "normal"
        self.speed_state = "normal"
        self.is_moving = False
        self.reactions = ROBOT_REACTIONS
        
        # Simulated hardware states
        self.tail_angle = 0
        self.head_angle = 0
        self.left_eye = "normal"
        self.right_eye = "normal"
        
        # Start background animation thread
        self.running = True
        self.animation_thread = threading.Thread(target=self._animation_loop)
        self.animation_thread.daemon = True
        self.animation_thread.start()
        
        print("🤖 Robot Controller initialized")
    
    def react_to_emotion(self, emotion, person_name=None):
        """
        React to detected emotion
        """
        self.current_emotion = emotion
        self.current_person = person_name
        
        if emotion in self.reactions:
            reaction = self.reactions[emotion]
            self.tail_wag_state = reaction["tail_wag"]
            self.speed_state = reaction["speed"]
            
            # Set eye expressions
            self._set_eye_expression(emotion)
            
            print(f"🤖 Reacting to {emotion} emotion - Tail: {self.tail_wag_state}, Speed: {self.speed_state}")
        else:
            print(f"🤖 No reaction defined for {emotion}")
    
    def _set_eye_expression(self, emotion):
        """
        Set eye expressions based on emotion
        """
        expressions = {
            "happy": ("wide", "wide"),
            "sad": ("droopy", "droopy"),
            "angry": ("narrow", "narrow"),
            "fear": ("wide", "wide"),
            "surprise": ("very_wide", "very_wide"),
            "neutral": ("normal", "normal")
        }
        
        if emotion in expressions:
            self.left_eye, self.right_eye = expressions[emotion]
    
    def _animation_loop(self):
        """
        Background loop for continuous animations
        """
        wag_direction = 1
        wag_speed_map = {
            "fast": 0.05,
            "normal": 0.1,
            "slow": 0.2,
            "none": None
        }
        
        while self.running:
            if self.tail_wag_state != "none":
                wag_speed = wag_speed_map.get(self.tail_wag_state, 0.1)
                
                if wag_speed:
                    # Wag tail animation
                    self.tail_angle += wag_direction * 15
                    if abs(self.tail_angle) > 45:
                        wag_direction *= -1
                    
                    # Simulate hardware update
                    # In real implementation, send commands to servo motors
                    
                    time.sleep(wag_speed)
            else:
                # Tail still
                self.tail_angle = 0
                time.sleep(0.1)
    
    def move_forward(self):
        """Move robot forward"""
        self.is_moving = True
        speed_map = {
            "fast": 1.0,
            "normal": 0.5,
            "slow": 0.2,
            "cautious": 0.1,
            "backward": -0.2,
            "stop": 0
        }
        
        speed = speed_map.get(self.speed_state, 0.5)
        print(f"🤖 Moving at speed: {speed}")
        
        # In real implementation, send commands to wheel motors
    
    def move_backward(self):
        """Move robot backward"""
        self.is_moving = True
        print("🤖 Moving backward")
    
    def stop(self):
        """Stop robot movement"""
        self.is_moving = False
        print("🤖 Stopped")
    
    def turn_head(self, angle):
        """Turn robot head to specified angle"""
        self.head_angle = max(-90, min(90, angle))
        print(f"🤖 Turning head to {self.head_angle} degrees")
    
    def get_status(self):
        """Get current robot status"""
        return {
            "emotion": self.current_emotion,
            "person": self.current_person,
            "tail_wag": self.tail_wag_state,
            "speed": self.speed_state,
            "tail_angle": self.tail_angle,
            "head_angle": self.head_angle,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "is_moving": self.is_moving
        }
    
    def cleanup(self):
        """Cleanup robot resources"""
        self.running = False
        if self.animation_thread.is_alive():
            self.animation_thread.join(timeout=1.0)
        print("🤖 Robot Controller cleaned up")