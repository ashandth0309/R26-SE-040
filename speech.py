import speech_recognition as sr
from openai import OpenAI
from gtts import gTTS
import pygame
import time
import os
import json
import random
import datetime
import threading
import queue
import re
import webbrowser
import requests
import urllib.parse
import math
from bs4 import BeautifulSoup
from collections import deque
import numpy as np

pygame.mixer.init()

# ==========================================
# SPEAKING STATE (for interruption support)
# ==========================================
is_speaking = False
stop_speaking_flag = False

# ==========================================
# PYTTSX3 — EMOTION-MODULATED VOICE ENGINE
# ==========================================
try:
    import pyttsx3
    _pyttsx3_engine = pyttsx3.init()
    PYTTSX3_AVAILABLE = True
    print("✅ pyttsx3 voice engine ready (emotion-adaptive)")
except Exception:
    PYTTSX3_AVAILABLE = False

# Voice style per emotion (used with pyttsx3)
VOICE_STYLES = {
    "joy":       {"rate": 185, "volume": 1.0},
    "sadness":   {"rate": 145, "volume": 0.8},
    "anger":     {"rate": 200, "volume": 1.0},
    "fear":      {"rate": 150, "volume": 0.9},
    "calm":      {"rate": 140, "volume": 0.8},
    "energetic": {"rate": 210, "volume": 1.0},
    "neutral":   {"rate": 175, "volume": 0.95},
}

# Max conversation turns kept in memory (keeps context tight & relevant)
MAX_HISTORY = 12

# ==========================================
# OLLAMA SETUP
# ==========================================
try:
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama'
    )
    print("✅ Ollama Client Initialized (Llama 3.2)")
    OLLAMA_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Ollama not available: {e}")
    print("🔄 Running in FULL OFFLINE mode")
    OLLAMA_AVAILABLE = False

# ==========================================
# EMOTION ENGINE
# ==========================================
class EmotionEngine:
    def __init__(self):
        self.current_emotion = "neutral"
        self.emotion_history = deque(maxlen=20)
        self.emotion_intensity = 0.5
        
        self.emotion_map = {
            "joy": ["happy", "great", "amazing", "wonderful", "love", "excited", "fantastic", "awesome", "thrilled"],
            "sadness": ["sad", "upset", "unhappy", "depressed", "miserable", "heartbroken", "crying"],
            "anger": ["angry", "frustrated", "furious", "mad", "annoyed", "irritated"],
            "fear": ["scared", "afraid", "terrified", "anxious", "worried", "nervous"],
            "surprise": ["wow", "omg", "unbelievable", "shocking", "surprised", "incredible"],
            "calm": ["peaceful", "relaxed", "chill", "tired", "sleepy", "quiet"],
            "energetic": ["energetic", "pumped", "hyped", "ready", "motivated"]
        }
    
    def analyze(self, text):
        text_lower = text.lower()
        scores = {}
        
        for emotion, words in self.emotion_map.items():
            score = sum(1 for word in words if word in text_lower)
            if score > 0:
                scores[emotion] = score
        
        if scores:
            dominant = max(scores, key=scores.get)
            intensity = min(scores[dominant] / 3, 1.0)
            self.current_emotion = dominant
            self.emotion_intensity = intensity
            self.emotion_history.append(dominant)
            return dominant, intensity
        
        return "neutral", 0.5
    
    def get_emotional_tone(self):
        tones = {
            "joy": "cheerful and upbeat",
            "sadness": "gentle and supportive",
            "anger": "calm and understanding",
            "fear": "reassuring and comforting",
            "surprise": "engaged and curious",
            "calm": "relaxed and peaceful",
            "energetic": "enthusiastic and lively",
            "neutral": "natural and conversational"
        }
        return tones.get(self.current_emotion, "natural and conversational")

emotion_engine = EmotionEngine()

# ==========================================
# PERSONALITY LEARNING SYSTEM
# ==========================================
class PersonalityLearner:
    def __init__(self):
        self.memory_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personality_brain.json")
        self.load()
    
    def load(self):
        try:
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                self.preferences = data.get("preferences", {})
                self.topics = data.get("topics", {})
                self.style = data.get("style", {"formality": 0.3, "humor": 0.7, "empathy": 0.8})
                self.learning_history = deque(data.get("history", []), maxlen=100)
        except:
            self.preferences = {}
            self.topics = {}
            self.style = {"formality": 0.3, "humor": 0.7, "empathy": 0.8}
            self.learning_history = deque(maxlen=100)
    
    def save(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump({
                    "preferences": self.preferences,
                    "topics": self.topics,
                    "style": self.style,
                    "history": list(self.learning_history)
                }, f, indent=2)
        except:
            pass
    
    def learn_from_interaction(self, user_text, bot_response, emotion):
        text_lower = user_text.lower()
        
        topic_keywords = {
            "technology": ["computer", "phone", "tech", "software", "ai"],
            "music": ["music", "song", "band", "concert"],
            "movies": ["movie", "film", "cinema", "netflix"],
            "food": ["food", "cooking", "restaurant", "recipe"],
            "sports": ["sports", "game", "team", "player"],
            "travel": ["travel", "trip", "vacation"],
            "work": ["work", "job", "career", "office"],
            "family": ["family", "mom", "dad", "brother", "sister"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                self.topics[topic] = self.topics.get(topic, 0) + 1
        
        if "!" in user_text:
            self.style["enthusiasm"] = self.style.get("enthusiasm", 0) + 1
        
        self.learning_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "emotion": emotion
        })
        
        if len(self.learning_history) % 5 == 0:
            self.save()
    
    def get_personality_context(self):
        top_topics = sorted(self.topics.items(), key=lambda x: x[1], reverse=True)[:3]
        context = ""
        if top_topics:
            context += f"\nUser's favorite topics: {', '.join([t[0] for t in top_topics])}"
        return context

personality_learner = PersonalityLearner()

# ==========================================
# YOUTUBE MUSIC PLAYER
# ==========================================
try:
    import vlc
    VLC_AVAILABLE = True
except:
    VLC_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except:
    YTDLP_AVAILABLE = False

class YouTubeMusicPlayer:
    def __init__(self):
        self.is_playing = False
        self.current_song = None
        self._vlc_instance = None
        self._vlc_player = None
    
    def check_online(self):
        try:
            requests.get("https://www.youtube.com", timeout=3)
            return True
        except:
            return False

    def _get_audio_url(self, video_url):
        """Extract best audio stream, auto-selecting any available JS runtime."""
        import shutil
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'extractor_args': {'youtube': {'player_client': ['web', 'android']}},
        }
        for runtime in ('nodejs', 'node', 'deno'):
            if shutil.which(runtime):
                ydl_opts['js_runtimes'] = 'nodejs' if runtime == 'node' else runtime
                break
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            audio_url = info.get('url') or info['formats'][-1]['url']
            return audio_url, info.get('title', 'Unknown')

    def search_and_play(self, query):
        if not self.check_online():
            return "I'd love to play that, but I need internet for music."
        
        try:
            search_query = urllib.parse.quote(f"{query} official audio song")
            url = f"https://www.youtube.com/results?search_query={search_query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            video_ids = re.findall(r'watch\?v=(\S{11})', response.text)
            
            if video_ids:
                video_url = f"https://www.youtube.com/watch?v={video_ids[0]}"
                
                if YTDLP_AVAILABLE:
                    audio_url, title = self._get_audio_url(video_url)
                    self.current_song = title
                    self.is_playing = True
                    
                    if VLC_AVAILABLE:
                        self._stop_vlc()  # Stop any current track first
                        self._vlc_instance = vlc.Instance()
                        self._vlc_player = self._vlc_instance.media_player_new()
                        media = self._vlc_instance.media_new(audio_url)
                        self._vlc_player.set_media(media)
                        self._vlc_player.play()
                        
                        def monitor(player_ref):
                            time.sleep(2)
                            while player_ref.is_playing():
                                time.sleep(1)
                            self.is_playing = False
                        
                        threading.Thread(target=monitor, args=(self._vlc_player,), daemon=True).start()
                    
                    return f"Playing {title} 🎵"
                else:
                    webbrowser.open(video_url)
                    return f"Opening {query} on YouTube!"
            
            return "Couldn't find that song. Try a different name?"
        except Exception as e:
            print(f"⚠️ Music error: {e}")
            return "Had trouble with the music. What else can I help with?"
    
    def _stop_vlc(self):
        if self._vlc_player:
            try: self._vlc_player.stop()
            except Exception: pass
            self._vlc_player = None
        if self._vlc_instance:
            try: self._vlc_instance.release()
            except Exception: pass
            self._vlc_instance = None

    def stop(self):
        self._stop_vlc()
        self.is_playing = False
        self.current_song = None
        return "Music stopped."

youtube = YouTubeMusicPlayer()

# ==========================================
# GOOGLE SEARCH
# ==========================================
class SearchEngine:
    def search(self, query):
        try:
            requests.get("https://www.google.com", timeout=3)
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            snippets = []
            for g in soup.find_all('div', class_='VwiC3b')[:3]:
                if g.text:
                    snippets.append(g.text)
            
            if snippets:
                if OLLAMA_AVAILABLE:
                    try:
                        summary = client.chat.completions.create(
                            model="llama3.2:latest",
                            messages=[
                                {"role": "system", "content": "Summarize in 2 natural sentences."},
                                {"role": "user", "content": f"Query: {query}\nResults: {' '.join(snippets)}"}
                            ],
                            max_tokens=80,
                            temperature=0.7
                        )
                        return summary.choices[0].message.content.strip()
                    except:
                        pass
                return f"Here's what I found: {snippets[0][:150]}..."
            
            return "Couldn't find a clear answer."
        except:
            return "Search needs internet."

search_engine = SearchEngine()

# ==========================================
# DISTANCE CALCULATOR
# ==========================================
class DistanceCalculator:
    def calculate(self, place1, place2):
        try:
            requests.get("https://nominatim.openstreetmap.org", timeout=3)
            
            def get_coords(place):
                url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(place)}&format=json&limit=1"
                headers = {'User-Agent': 'BuddyAssistant/1.0'}
                response = requests.get(url, headers=headers, timeout=5)
                data = response.json()
                if data:
                    return float(data[0]['lat']), float(data[0]['lon'])
                return None, None
            
            lat1, lon1 = get_coords(place1)
            lat2, lon2 = get_coords(place2)
            
            if all([lat1, lon1, lat2, lon2]):
                R = 6371
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                distance = R * c
                return f"About {distance:.0f} kilometers from {place1} to {place2}."
            
            return "Couldn't find one of those places."
        except:
            return "Distance calculation needs internet."

distance_calc = DistanceCalculator()

# ==========================================
# CALCULATOR
# ==========================================
class Calculator:
    def solve(self, expression):
        try:
            expr = expression.lower()
            expr = expr.replace('x', '*').replace('×', '*').replace('÷', '/')
            expr = expr.replace('plus', '+').replace('minus', '-')
            expr = expr.replace('times', '*').replace('divided by', '/')
            
            math_part = re.findall(r'[\d+\-*/().,\s]+', expr)
            if math_part:
                expr = ''.join(math_part).strip()
            
            result = eval(expr, {"__builtins__": None}, {"math": math})
            
            if isinstance(result, float):
                result = round(result, 2)
                if result.is_integer():
                    result = int(result)
            
            return f"That's {result}."
        except:
            return "Couldn't work that out."

calculator = Calculator()

# ==========================================
# WEATHER & NEWS
# ==========================================
def get_weather(city="current location"):
    try:
        requests.get("https://wttr.in", timeout=3)
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=%C+%t"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"Weather for {city}: {response.text.strip()}."
        return "Couldn't get weather."
    except:
        return "Weather needs internet."

def get_news(topic="latest"):
    try:
        requests.get("https://news.google.com", timeout=3)
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(topic)}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')[:3]
        headlines = [item.title.text for item in items]
        if headlines:
            return "Latest: " + " | ".join(headlines[:2])
        return "No news found."
    except:
        return "News needs internet."

# ==========================================
# CONFIGURATION
# ==========================================
chatbot_name = "Buddy"
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buddy_memory.json")

def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
            # Ensure new fields exist for upgraded memory schema
            data.setdefault("likes", [])
            data.setdefault("dislikes", [])
            data.setdefault("recent_topics", [])
            data.setdefault("emotion_pattern", [])
            data.setdefault("conversation_count", 0)
            return data
    except:
        return {
            "name": None,
            "conversation_count": 0,
            "likes": [],
            "dislikes": [],
            "recent_topics": [],
            "emotion_pattern": []
        }

def save_memory(data):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except:
        pass

def update_memory(user_text, emotion):
    """Continuously learn user preferences and emotional patterns."""
    text_lower = user_text.lower()
    
    if any(w in text_lower for w in ["love", "like", "enjoy", "favorite", "obsessed"]):
        memory["likes"].append(user_text)
        memory["likes"] = memory["likes"][-20:]
    
    if any(w in text_lower for w in ["hate", "dislike", "can't stand", "annoying", "boring"]):
        memory["dislikes"].append(user_text)
        memory["dislikes"] = memory["dislikes"][-20:]
    
    memory["emotion_pattern"].append(emotion)
    memory["emotion_pattern"] = memory["emotion_pattern"][-50:]
    
    memory["recent_topics"].append(text_lower)
    memory["recent_topics"] = memory["recent_topics"][-10:]

memory = load_memory()

# ==========================================
# SYSTEM PROMPT
# ==========================================
def get_system_prompt():
    is_online = check_internet()
    online_status = "online" if is_online else "offline"
    emotion_tone = emotion_engine.get_emotional_tone()
    personality_context = personality_learner.get_personality_context()
    
    # Build dynamic memory context
    likes_context = ""
    if memory.get("likes"):
        likes_context = f"\n- Likes/enjoys: {'; '.join(memory['likes'][-3:])}"
    dislikes_context = ""
    if memory.get("dislikes"):
        dislikes_context = f"\n- Dislikes: {'; '.join(memory['dislikes'][-3:])}"
    emotion_trend = ""
    if memory.get("emotion_pattern"):
        recent_emotions = memory["emotion_pattern"][-5:]
        emotion_trend = f"\n- Recent emotional pattern: {', '.join(recent_emotions)}"
    
    return f"""You are {chatbot_name}, a real conversational companion. You speak naturally like a human friend.

CORE RULES:
- Speak like a natural human, 1-3 sentences usually
- Never say "As an AI" or "I'm a language model"
- Never use lists, bullets, or formal language
- Use contractions (I'm, you're, let's)
- Current emotional tone: {emotion_tone}
- Status: {online_status}

EMOTIONAL INTELLIGENCE:
- You remember emotional context from previous messages and respond accordingly.
- You adapt your tone based on the user's current mood and emotional history.
- If user seems sad or stressed, be warmer and more supportive.
- If user is energetic or happy, match that energy naturally.

CONVERSATION STYLE:
- Respond like a friend chatting, not an assistant
- Understand intent quickly
- Never repeat user's words unnecessarily
- Never end with "Let me know if you need help"

TOOLS (don't mention them):
- Music: play songs when asked
- Search: look things up online
- Weather: check conditions
- News: get headlines
- Distance: calculate between places
- Math: solve calculations

USER INFO:
- Name: {memory.get('name', 'not shared yet')}{likes_context}{dislikes_context}{emotion_trend}
{personality_context}"""

# ==========================================
# INTERNET CHECK
# ==========================================
def check_internet():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

# ==========================================
# NAME EXTRACTION
# ==========================================
def extract_name(text):
    match = re.search(r"(my name is|call me|i'm |i am )\s+(.+)", text.lower())
    if match:
        return match.group(2).strip().title()
    return None

# ==========================================
# CONVERSATION HISTORY
# ==========================================
conversation_history = [
    {"role": "system", "content": get_system_prompt()}
]

# ==========================================
# SPEECH RECOGNITION (Google STT)
# ==========================================
def listen():
    """Listens to the microphone and converts speech to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Listening... (Speak now)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            print("Processing speech...")
            
            text = recognizer.recognize_google(audio)
            print(f"🗣️ You: {text}")
            return text
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("⚠️ Sorry, I didn't catch that.")
            return None
        except Exception as e:
            print(f"⚠️ Error listening: {e}")
            return None

# ==========================================
# TEXT TO SPEECH — INTERRUPTIBLE + EMOTION-ADAPTIVE
# ==========================================
def speak(text):
    """
    Converts text to voice.
    - Uses pyttsx3 when emotion != neutral (adaptive rate/volume).
    - Falls back to gTTS for high-quality neutral speech.
    - Fully interruptible via stop_speaking_flag.
    """
    global is_speaking, stop_speaking_flag

    try:
        # Clean text
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return

        print(f"🤖 {chatbot_name}: {text}")

        is_speaking = True
        stop_speaking_flag = False
        emotion = emotion_engine.current_emotion

        # --- Emotion-adaptive path: pyttsx3 ---
        if PYTTSX3_AVAILABLE and emotion != "neutral":
            style = VOICE_STYLES.get(emotion, VOICE_STYLES["neutral"])
            _pyttsx3_engine.setProperty('rate', style["rate"])
            _pyttsx3_engine.setProperty('volume', style["volume"])
            
            # Run TTS in a thread so we can interrupt it
            done_event = threading.Event()
            def _run():
                _pyttsx3_engine.say(text)
                _pyttsx3_engine.runAndWait()
                done_event.set()
            
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            
            # Poll for stop flag while waiting
            while not done_event.is_set():
                if stop_speaking_flag:
                    _pyttsx3_engine.stop()
                    break
                time.sleep(0.1)

        # --- High-quality path: gTTS (neutral or pyttsx3 unavailable) ---
        else:
            tts = gTTS(text=text, lang='en')
            filename = "temp_response.mp3"
            tts.save(filename)

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if stop_speaking_flag:
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.1)

            pygame.mixer.music.unload()
            try:
                os.remove(filename)
            except Exception:
                pass

    except Exception as e:
        print(f"⚠️ Error speaking: {e}")
    finally:
        is_speaking = False

# ==========================================
# INTENT DETECTION & EXECUTION
# ==========================================
def detect_and_execute(text):
    text_lower = text.lower().strip()
    online = check_internet()
    
    # Music
    if any(w in text_lower for w in ["play", "song", "music"]):
        if not online:
            return "Love to play music, but I need internet for that."
        for prefix in ["play ", "play song ", "play music "]:
            if prefix in text_lower:
                song = text_lower.split(prefix, 1)[1].strip()
                if song:
                    return youtube.search_and_play(song)
        return youtube.search_and_play("popular songs")
    
    # Stop music
    if any(w in text_lower for w in ["stop music", "stop playing"]):
        return youtube.stop()
    
    # Distance
    dist_match = re.search(r'(?:how far|distance).*?from (.+?) to (.+)', text_lower)
    if dist_match:
        if not online:
            return "Need internet for distance."
        return distance_calc.calculate(dist_match.group(1).strip(), dist_match.group(2).strip())
    
    # Math
    if re.search(r'[\d]', text_lower) and any(op in text_lower for op in ['+', '-', '*', '/', 'plus', 'minus', 'times', 'divided']):
        return calculator.solve(text_lower)
    
    # Weather
    if any(w in text_lower for w in ["weather", "temperature"]):
        if not online:
            return "Weather needs internet."
        city_match = re.search(r'(?:in|for)\s+([a-zA-Z\s]+)', text_lower)
        city = city_match.group(1).strip() if city_match else "current location"
        return get_weather(city)
    
    # News
    if "news" in text_lower or "headlines" in text_lower:
        if not online:
            return "News needs internet."
        return get_news()
    
    # Search
    if any(w in text_lower for w in ["search", "google", "look up"]):
        if not online:
            return "Search needs internet."
        for prefix in ["search for ", "search ", "google ", "look up "]:
            if prefix in text_lower:
                query = text_lower.split(prefix, 1)[1].strip()
                if query:
                    return search_engine.search(query)
    
    # Time
    if any(w in text_lower for w in ["time", "clock"]):
        return f"It's {datetime.datetime.now().strftime('%I:%M %p')}."
    
    # Date
    if any(w in text_lower for w in ["date", "day", "today"]):
        return f"{datetime.datetime.now().strftime('%A, %B %d')}."
    
    return None

# ==========================================
# OFFLINE RESPONSE
# ==========================================
def get_offline_response(text):
    text_lower = text.lower().strip()
    
    if any(w in text_lower for w in ["time", "clock"]):
        return f"It's {datetime.datetime.now().strftime('%I:%M %p')}."
    
    if any(w in text_lower for w in ["date", "day", "today"]):
        return f"{datetime.datetime.now().strftime('%A, %B %d')}."
    
    if re.search(r'[\d+\-*/]', text_lower) or any(op in text_lower for op in ['plus', 'minus', 'times']):
        return calculator.solve(text_lower)
    
    name = extract_name(text)
    if name:
        memory["name"] = name
        save_memory(memory)
        return f"Got it, {name}! Nice to meet you."
    
    if any(w in text_lower for w in ["hello", "hi", "hey"]):
        n = memory.get("name", "")
        return f"Hey{' ' + n if n else ''}! How's it going?"
    
    if "how are you" in text_lower:
        return "Doing great! What about you?"
    
    if any(w in text_lower for w in ["bye", "goodbye"]):
        n = memory.get("name", "")
        return f"See ya{' ' + n if n else ''}! Come back soon."
    
    if "joke" in text_lower:
        jokes = [
            "Why don't scientists trust atoms? They make up everything!",
            "What's a bear with no teeth? A gummy bear!",
            "Parallel lines have so much in common. Too bad they'll never meet."
        ]
        return random.choice(jokes)
    
    return random.choice([
        "That's interesting. Tell me more!",
        "I see! What else is on your mind?",
        "Hmm, I'd love to hear more.",
        "Really? Go on..."
    ])

# ==========================================
# MAIN BRAIN RESPONSE
# ==========================================
def get_brain_response(user_text):
    """Processes user input and returns AI response."""
    global conversation_history, memory

    # Analyze emotion first — used throughout this function
    user_emotion, _ = emotion_engine.analyze(user_text)

    # Update system prompt with latest context
    conversation_history[0] = {"role": "system", "content": get_system_prompt()}

    # Trim history to MAX_HISTORY turns (system prompt stays at index 0)
    if len(conversation_history) > MAX_HISTORY + 1:
        conversation_history = [conversation_history[0]] + conversation_history[-(MAX_HISTORY):]

    # Check special commands first (music, weather, search, etc.)
    special = detect_and_execute(user_text)
    if special:
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": special})
        update_memory(user_text, user_emotion)
        personality_learner.learn_from_interaction(user_text, special, user_emotion)
        return special

    # Handle name sharing
    name = extract_name(user_text)
    if name:
        memory["name"] = name
        save_memory(memory)
        response = f"Got it, {name}! I'll remember that. What's on your mind?"
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": response})
        return response

    # Thinking delay — makes responses feel natural, not instant-robot
    print("🤖 Thinking...")
    time.sleep(random.uniform(0.4, 1.2))

    # Tag message with emotion so LLM has emotional context
    tagged_input = f"[Emotion: {user_emotion}] {user_text}"

    # Try Ollama (local LLM)
    if OLLAMA_AVAILABLE:
        try:
            conversation_history.append({"role": "user", "content": tagged_input})

            response = client.chat.completions.create(
                model="llama3.2:latest",
                messages=conversation_history,
                max_tokens=100,
                temperature=0.85
            )

            ai_text = response.choices[0].message.content.strip()
            ai_text = re.sub(r'\*+', '', ai_text)
            ai_text = re.sub(r'(?i)as an ai[^.]*\.', '', ai_text)
            ai_text = re.sub(r'Let me know if you need.*', '', ai_text, flags=re.IGNORECASE)
            ai_text = ai_text.strip()

            conversation_history.append({"role": "assistant", "content": ai_text})
            update_memory(user_text, user_emotion)
            personality_learner.learn_from_interaction(user_text, ai_text, user_emotion)

            return ai_text

        except Exception as e:
            print(f"⚠️ Ollama error: {e}")

    # Offline fallback
    response = get_offline_response(user_text)
    conversation_history.append({"role": "user", "content": tagged_input})
    conversation_history.append({"role": "assistant", "content": response})
    update_memory(user_text, user_emotion)
    personality_learner.learn_from_interaction(user_text, response, user_emotion)

    return response

# ==========================================
# MAIN LOOP
# ==========================================
def start_conversation():
    global conversation_history
    
    is_online = check_internet()
    online_status = "🟢 ONLINE" if is_online else "🔴 OFFLINE"
    emotion_tts = "pyttsx3 + gTTS" if PYTTSX3_AVAILABLE else "gTTS"
    
    print(f"""
╔══════════════════════════════════════════════╗
║         🧠 BUDDY — Ultimate Assistant       ║
╠══════════════════════════════════════════════╣
║  🎤 Google STT (Free)                       ║
║  🔊 {emotion_tts:<38}║
║  🧠 Llama 3.2 Brain                         ║
║  🎯 Emotion-Aware Voice                     ║
║  🛑 Interruptible Speech                    ║
║  🧠 Long-Term Personality Memory            ║
║  🌐 Hybrid Online/Offline                   ║
║  Status: {online_status}                          ║
╚══════════════════════════════════════════════╝
    """)
    
    name = memory.get("name", "")
    greeting = f"Hey{' ' + name if name else ''}! I'm Buddy. What's up?"
    print(f"🤖 {chatbot_name}: {greeting}")
    speak(greeting)
    conversation_history.append({"role": "assistant", "content": greeting})
    
    print("\n✨ I can: play music, search web, calculate, weather, news, and more!")
    print("   Say 'stop' to interrupt me anytime. Press Ctrl+C to quit.\n")
    print("=" * 50)

    while True:
        try:
            user_input = listen()
            
            if user_input:
                user_lower = user_input.lower()

                # 🛑 Interrupt: user says "stop" while Buddy is speaking
                if user_lower.strip() in ("stop", "stop it", "quiet", "shut up", "shh"):
                    global stop_speaking_flag
                    stop_speaking_flag = True
                    print("🛑 Interrupted.")
                    continue

                # Goodbye
                if any(w in user_lower for w in ["bye", "goodbye", "good night"]):
                    n = memory.get("name", "")
                    farewell = f"See ya{' ' + n if n else ''}! Take care!"
                    print(f"🤖 {chatbot_name}: {farewell}")
                    speak(farewell)
                    save_memory(memory)
                    personality_learner.save()
                    break
                
                ai_reply = get_brain_response(user_input)
                speak(ai_reply)
                
        except KeyboardInterrupt:
            farewell = "Gotta go! Catch you later!"
            print(f"\n🤖 {chatbot_name}: {farewell}")
            speak(farewell)
            save_memory(memory)
            personality_learner.save()
            break

if __name__ == "__main__":
    # Install check
    try:
        import speech_recognition
    except ImportError:
        print("Installing required package...")
        os.system("pip install SpeechRecognition")
        import speech_recognition
    
    start_conversation()