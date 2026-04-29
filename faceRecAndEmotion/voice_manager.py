"""
Voice Manager for Text-to-Speech Output
Windows-compatible version using pygame
"""

from gtts import gTTS
import os
import tempfile
import threading
import time
import pygame
import io

class VoiceManager:
    def __init__(self):
        self.language = 'en'
        self.speed = 1.0
        self.is_speaking = False
        self.speech_queue = []
        self.last_greeting = {}
        self.greeting_cooldown = 60  # Don't greet same person within 60 seconds
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init()
            print("✅ Pygame mixer initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize pygame mixer: {e}")
    
    def speak(self, text):
        """
        Convert text to speech and play it using pygame
        """
        def speak_thread():
            self.is_speaking = True
            temp_filename = None
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_filename = fp.name
                
                # Generate speech
                tts = gTTS(text=text, lang=self.language, slow=(self.speed < 1.0))
                tts.save(temp_filename)
                
                # Load and play with pygame
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in speech: {e}")
            
            finally:
                # Clean up temp file
                if temp_filename and os.path.exists(temp_filename):
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass
                
                self.is_speaking = False
        
        # Start speaking in separate thread
        thread = threading.Thread(target=speak_thread)
        thread.daemon = True
        thread.start()
    
    def speak_simple(self, text):
        """
        Alternative method using Windows SAPI (if pygame fails)
        """
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        except ImportError:
            print(f"Would say: {text}")
            # Fallback to print
        except Exception as e:
            print(f"Speech error: {e}")
    
    def greet_person(self, name, emotion):
        """
        Generate greeting based on person and emotion
        """
        current_time = time.time()
        
        # Check cooldown
        if name in self.last_greeting:
            if current_time - self.last_greeting[name] < self.greeting_cooldown:
                return
        
        # Update last greeting time
        self.last_greeting[name] = current_time
        
        # Generate appropriate greeting
        greetings = {
            "happy": [
                f"Hello {name}! You look happy today!",
                f"Hi {name}! Great to see you smiling!",
                f"Hey {name}! You're in a good mood!"
            ],
            "sad": [
                f"Hello {name}. Is everything okay?",
                f"Hi {name}. I hope you feel better soon.",
                f"Hey {name}. I'm here if you need a friend."
            ],
            "angry": [
                f"Hello {name}. Let's take a deep breath.",
                f"Hi {name}. Everything will be okay.",
                f"Hey {name}. I'm here to help you relax."
            ],
            "surprise": [
                f"Wow {name}! You look surprised!",
                f"Hi {name}! Did something exciting happen?",
                f"Hey {name}! You seem amazed!"
            ],
            "neutral": [
                f"Hello {name}.",
                f"Hi {name}, nice to see you.",
                f"Hey {name}, how are you?"
            ]
        }
        
        import random
        if emotion in greetings:
            greeting = random.choice(greetings[emotion])
        else:
            greeting = f"Hello {name}!"
        
        # Try pygame first, fallback to SAPI, then print
        try:
            self.speak(greeting)
        except:
            try:
                self.speak_simple(greeting)
            except:
                print(f"🔊 {greeting}")
    
    def greet_unknown(self):
        """
        Greeting for unknown person
        """
        messages = [
            "Hello! I don't think we've met before.",
            "Hi there! You look new around here.",
            "Welcome! I don't recognize you."
        ]
        
        import random
        greeting = random.choice(messages)
        
        try:
            self.speak(greeting)
        except:
            try:
                self.speak_simple(greeting)
            except:
                print(f"🔊 {greeting}")
    
    def announce_emotion(self, emotion):
        """
        Announce detected emotion
        """
        announcements = {
            "happy": "You seem happy!",
            "sad": "You look a bit sad.",
            "angry": "You appear angry.",
            "surprise": "You look surprised!",
            "neutral": "You seem neutral."
        }
        
        if emotion in announcements:
            text = announcements[emotion]
            try:
                self.speak(text)
            except:
                try:
                    self.speak_simple(text)
                except:
                    print(f"🔊 {text}")