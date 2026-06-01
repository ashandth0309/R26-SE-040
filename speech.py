import ast
import datetime
import json
import math
import os
import random
import re
import time
import urllib.parse
import webbrowser
from collections import deque

import pygame
import requests
import speech_recognition as sr

try:
    from faceRecAndEmotion.robot_bridge import handle_robot_voice_command
except Exception:
    def handle_robot_voice_command(text):
        return None
    
from bs4 import BeautifulSoup
from gtts import gTTS
from openai import OpenAI

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import yt_dlp
except Exception:
    yt_dlp = None

try:
    import vlc
except Exception:
    vlc = None

try:
    from ml_brain import predict_intent
except Exception:
    def predict_intent(text, threshold=0.60):
        return "unknown"


APP_NAME = "Buddy"
MEMORY_FILE = "buddy_memory.json"
PERSONALITY_FILE = "personality_brain.json"

OLLAMA_BASE_URL = "http://localhost:11434/v1"

FAST_MODEL = "llama3.2:1b"
SMART_MODEL = "llama3.2:latest"
OLLAMA_MODEL = FAST_MODEL

TEMP_AUDIO_FILE = "temp_response.mp3"

LISTEN_TIMEOUT = 8
PHRASE_TIME_LIMIT = 25
OLLAMA_MAX_TOKENS = 100
OLLAMA_TEMPERATURE = 0.35

pygame.mixer.init()

is_speaking = False
stop_speaking_flag = False

INTERNET_CACHE = {"value": None, "time": 0}
OLLAMA_CACHE = {"value": None, "time": 0}


# ============================================================
# STATUS CHECKS
# ============================================================

def check_internet():
    now = time.time()

    if INTERNET_CACHE["value"] is not None and now - INTERNET_CACHE["time"] < 20:
        return INTERNET_CACHE["value"]

    try:
        requests.get("https://www.google.com", timeout=1)
        INTERNET_CACHE["value"] = True
    except Exception:
        INTERNET_CACHE["value"] = False

    INTERNET_CACHE["time"] = now
    return INTERNET_CACHE["value"]


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_json_load(path, default):
    try:
        if not os.path.exists(path):
            return default

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return default


def safe_json_save(path, data):
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    except Exception as error:
        print(f"Could not save {path}: {error}")


def clean_speech_text(text):
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):
    if not text:
        return "en"

    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"

    text_lower = text.lower().strip()

    strong_english_patterns = [
        r"\bwhat\b", r"\bwhy\b", r"\bwhen\b", r"\bwhere\b", r"\bwho\b",
        r"\bhow\b", r"\bcan you\b", r"\bcould you\b", r"\btell me\b",
        r"\bexplain\b", r"\bhelp me\b", r"\bdo you\b", r"\bare you\b",
        r"\bis it\b", r"\bi am\b", r"\bi want\b", r"\bplease\b",
        r"\bthanks\b", r"\bthank you\b", r"\bhello\b", r"\bhi\b",
        r"\bhey\b", r"\bmeaning\b", r"\bscience\b", r"\bquestion\b"
    ]

    for pattern in strong_english_patterns:
        if re.search(pattern, text_lower):
            return "en"

    tamil_english_words = [
        "vanakkam", "naan", "nan", "neenga", "nee", "ungal", "unakku",
        "enna", "enakku", "epdi", "eppadi", "saptiya", "saaptiya",
        "seri", "sari", "illa", "illai", "irukku", "venum", "pannu",
        "pannunga", "pesu", "pesunga", "tamil", "puriyudha", "puriyuma",
        "sollu", "sollunga", "kelu", "kelunga", "nalla", "romba",
        "konjam", "ippo", "ippove", "naalaiku", "inniku", "yosichu",
        "padikanum", "padikka", "exam ku", "sapadu", "saapadu",
        "thanni", "velai", "veedu", "amma", "appa"
    ]

    if any(word in text_lower for word in tamil_english_words):
        return "ta"

    return "en"


# ============================================================
# MEMORY
# ============================================================

def load_memory():
    default = {
        "name": None,
        "conversation_count": 0,
        "likes": [],
        "dislikes": [],
        "recent_topics": [],
        "emotion_pattern": [],
        "last_user_question": "",
        "last_bot_answer": "",
        "last_topic": "",
        "saved_questions": []
    }

    data = safe_json_load(MEMORY_FILE, default)

    for key, value in default.items():
        data.setdefault(key, value)

    return data


memory = load_memory()


def save_memory():
    safe_json_save(MEMORY_FILE, memory)


# ============================================================
# EMOTION ENGINE
# ============================================================

class EmotionEngine:
    def __init__(self):
        self.current_emotion = "neutral"
        self.emotion_history = deque(maxlen=20)
        self.emotion_intensity = 0.5

        self.emotion_map = {
            "joy": [
                "happy", "great", "amazing", "wonderful", "love", "excited",
                "fantastic", "awesome", "thrilled", "good", "sandhosham",
                "santhosham", "nalla"
            ],
            "sadness": [
                "sad", "upset", "unhappy", "depressed", "miserable",
                "heartbroken", "crying", "lonely", "kashtam", "sogam"
            ],
            "anger": [
                "angry", "frustrated", "furious", "mad", "annoyed",
                "irritated", "kobam"
            ],
            "fear": [
                "scared", "afraid", "terrified", "anxious", "worried",
                "nervous", "bayam"
            ],
            "calm": [
                "peaceful", "relaxed", "chill", "tired", "sleepy",
                "quiet", "amaidhi"
            ],
            "energetic": [
                "energetic", "pumped", "hyped", "ready", "motivated"
            ]
        }

    def analyze(self, text):
        text_lower = text.lower()
        scores = {}

        for emotion, keywords in self.emotion_map.items():
            score = sum(1 for word in keywords if word in text_lower)
            if score:
                scores[emotion] = score

        if not scores:
            self.current_emotion = "neutral"
            self.emotion_intensity = 0.5
            self.emotion_history.append("neutral")
            return "neutral", 0.5

        dominant = max(scores, key=scores.get)
        intensity = min(scores[dominant] / 3, 1.0)

        self.current_emotion = dominant
        self.emotion_intensity = intensity
        self.emotion_history.append(dominant)

        return dominant, intensity

    def get_tone(self):
        tones = {
            "joy": "cheerful and upbeat",
            "sadness": "gentle and supportive",
            "anger": "calm and understanding",
            "fear": "reassuring and comforting",
            "calm": "relaxed and peaceful",
            "energetic": "enthusiastic and lively",
            "neutral": "natural and conversational"
        }
        return tones.get(self.current_emotion, "natural and conversational")


emotion_engine = EmotionEngine()


# ============================================================
# PERSONALITY LEARNING
# ============================================================

class PersonalityLearner:
    def __init__(self):
        self.data = safe_json_load(
            PERSONALITY_FILE,
            {
                "preferences": {},
                "topics": {},
                "style": {
                    "formality": 0.3,
                    "humor": 0.7,
                    "empathy": 0.8,
                    "conciseness": 0.6
                },
                "history": []
            }
        )

        self.data.setdefault("preferences", {})
        self.data.setdefault("topics", {})
        self.data.setdefault("style", {})
        self.data.setdefault("history", [])

    def learn(self, user_text, bot_response, emotion):
        text = user_text.lower()

        topic_keywords = {
            "technology": ["computer", "phone", "tech", "software", "ai", "code", "python"],
            "music": ["music", "song", "band", "concert", "paatu"],
            "movies": ["movie", "film", "cinema", "netflix", "padam"],
            "food": ["food", "cooking", "restaurant", "recipe", "sapadu", "saapadu"],
            "sports": ["sports", "game", "team", "player"],
            "travel": ["travel", "trip", "vacation", "payanam"],
            "work": ["work", "job", "career", "office", "velai"],
            "family": ["family", "mom", "dad", "brother", "sister", "amma", "appa"],
            "study": ["exam", "study", "school", "college", "maths", "assignment", "padikka"],
            "science": ["science", "photosynthesis", "plant", "oxygen", "biology", "physics", "chemistry"]
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                self.data["topics"][topic] = self.data["topics"].get(topic, 0) + 1

        self.data["history"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "emotion": emotion
        })

        self.data["history"] = self.data["history"][-100:]

        if len(self.data["history"]) % 5 == 0:
            self.save()

    def get_context(self):
        topics = self.data.get("topics", {})
        top_topics = sorted(topics.items(), key=lambda item: item[1], reverse=True)[:3]

        if not top_topics:
            return ""

        topic_names = ", ".join(topic for topic, count in top_topics)
        return f"\nUser's recent topic interests: {topic_names}"

    def save(self):
        safe_json_save(PERSONALITY_FILE, self.data)


personality_learner = PersonalityLearner()


# ============================================================
# OLLAMA SETUP
# ============================================================

def create_ollama_client():
    try:
        return OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"
        )
    except Exception as error:
        print(f"Ollama client failed: {error}")
        return None


ollama_client = create_ollama_client()


def ollama_available():
    now = time.time()

    if OLLAMA_CACHE["value"] is not None and now - OLLAMA_CACHE["time"] < 20:
        return OLLAMA_CACHE["value"]

    if ollama_client is None:
        OLLAMA_CACHE["value"] = False
        OLLAMA_CACHE["time"] = now
        return False

    try:
        requests.get("http://localhost:11434", timeout=0.5)
        OLLAMA_CACHE["value"] = True
    except Exception:
        OLLAMA_CACHE["value"] = False

    OLLAMA_CACHE["time"] = now
    return OLLAMA_CACHE["value"]


# ============================================================
# NAME EXTRACTION
# ============================================================

def extract_name(text):
    patterns = [
        r"\bmy name is\s+(.+)",
        r"\bcall me\s+(.+)",
        r"\bi am\s+(.+)",
        r"\bi'm\s+(.+)",
        r"\bஎன் பெயர்\s+(.+)",
        r"\bஎன்னை\s+(.+?)\s+என்று கூப்பிடு",
        r"\bஎன்னை\s+(.+?)\s+னு கூப்பிடு"
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower().strip())

        if match:
            name = match.group(1).strip()
            name = re.sub(r"[^\w\s\u0B80-\u0BFF]", "", name).strip()

            bad_names = [
                "good", "fine", "okay", "ok", "happy", "sad", "ready",
                "going", "leaving", "asking", "here"
            ]

            if name and name.lower() not in bad_names:
                return name.title()

    return None


def update_memory(user_text, emotion):
    text = user_text.lower()

    memory["conversation_count"] = memory.get("conversation_count", 0) + 1

    if any(word in text for word in ["love", "like", "enjoy", "favorite", "pidikkum", "பிடிக்கும்"]):
        memory["likes"].append(user_text)
        memory["likes"] = memory["likes"][-20:]

    if any(word in text for word in ["hate", "dislike", "can't stand", "boring", "pidikkala", "பிடிக்கல"]):
        memory["dislikes"].append(user_text)
        memory["dislikes"] = memory["dislikes"][-20:]

    memory["recent_topics"].append(text)
    memory["recent_topics"] = memory["recent_topics"][-10:]

    memory["emotion_pattern"].append(emotion)
    memory["emotion_pattern"] = memory["emotion_pattern"][-50:]


# ============================================================
# FOLLOW-UP MEMORY
# ============================================================

def is_follow_up_question(text):
    text_lower = text.lower().strip()

    follow_up_patterns = [
        "explain that",
        "explain it",
        "explain this",
        "explain more",
        "explain more dee",
        "explain more deeply",
        "can you explain more",
        "can you explain more dee",
        "can you explain that",
        "can you explain it",
        "tell me more",
        "tell more",
        "more about that",
        "more about it",
        "what about that",
        "briefly explain",
        "explain briefly",
        "explain that briefly",
        "make it short",
        "say it simply",
        "what was that",
        "what is that",
        "why is that",
        "how is that",
        "continue that",
        "continue it",
        "go deeper",
        "go deep",
        "detail it",
        "details",
        "in detail",
        "what was the question",
        "what was the science question",
        "what question did i ask",
        "what did i ask before",
        "previous question",
        "last question",
        "simple ah sollu",
        "adha explain pannu",
        "அதை explain பண்ணு",
        "அதை விளக்கு",
        "சுருக்கமா சொல்லு",
        "brief ah sollu",
        "detail ah sollu",
        "more ah sollu"
    ]

    if any(pattern in text_lower for pattern in follow_up_patterns):
        return True

    short_follow_words = ["more", "brief", "briefly", "detail", "deeply", "again"]
    words = text_lower.split()

    if len(words) <= 6 and any(word in words for word in short_follow_words):
        return True

    return False


def is_memory_recall_question(text):
    text_lower = text.lower().strip()

    recall_patterns = [
        "what did i ask",
        "what was the question",
        "what was the science question",
        "what question did i ask",
        "what did i ask before",
        "previous question",
        "last question",
        "do you remember what i asked",
        "what i asked you before"
    ]

    return any(pattern in text_lower for pattern in recall_patterns)


def is_brief_request(text):
    text_lower = text.lower().strip()

    brief_words = [
        "brief", "briefly", "short", "simple", "simply", "summarize",
        "summary", "சுருக்கமா", "short ah", "simple ah", "brief ah"
    ]

    return any(word in text_lower for word in brief_words)


def is_detailed_request(text):
    text_lower = text.lower().strip()

    detail_words = [
        "detail", "detailed", "deeply", "explain deeply", "step by step",
        "full explanation", "proper explanation", "clearly explain",
        "விவரமா", "detail ah", "step by step"
    ]

    return any(word in text_lower for word in detail_words)


def extract_simple_topic(text):
    text_lower = text.lower().strip()

    remove_phrases = [
        "what is", "what are", "who is", "tell me about", "explain",
        "can you explain", "can you tell me about", "think and tell",
        "briefly", "in detail", "step by step", "please", "meaning of",
        "what the meaning of", "what is the meaning of"
    ]

    topic = text_lower

    for phrase in remove_phrases:
        topic = topic.replace(phrase, "")

    topic = re.sub(r"[^a-zA-Z0-9\u0B80-\u0BFF\s]", "", topic)
    topic = re.sub(r"\s+", " ", topic).strip()

    return topic[:100]


def build_contextual_user_text(user_text):
    if not is_follow_up_question(user_text) and not is_memory_recall_question(user_text):
        return user_text

    last_question = memory.get("last_user_question", "")
    last_answer = memory.get("last_bot_answer", "")
    last_topic = memory.get("last_topic", "")

    if not last_question and not last_answer and not last_topic:
        return user_text

    return f"""
The user is asking a follow-up or recall question.

Previous topic: {last_topic}
Previous user question: {last_question}
Previous Buddy answer: {last_answer}

Current user question: {user_text}

Important:
- Do not treat the follow-up as a new topic.
- Words like "that", "it", "more", "briefly", and misheard words like "Dee" refer to the previous topic.
- If the user asks what question they asked before, answer using the previous user question exactly.
- Otherwise, continue explaining the previous topic.
""".strip()


def save_last_interaction(user_text, bot_response):
    lower = user_text.lower().strip()

    dont_save_as_topic = [
        "hi", "hello", "hey", "bye", "goodbye", "thanks", "thank you",
        "stop", "stop it", "quiet", "shut up", "do you know my name",
        "do you remember my name", "what is my name", "tell me my name"
    ]

    if lower in dont_save_as_topic:
        return

    if is_memory_recall_question(user_text):
        return

    if is_follow_up_question(user_text):
        memory["last_bot_answer"] = bot_response
        save_memory()
        return

    topic = extract_simple_topic(user_text)

    memory["last_user_question"] = user_text
    memory["last_bot_answer"] = bot_response
    memory["last_topic"] = topic

    saved_questions = memory.get("saved_questions", [])
    saved_questions.append({
        "time": datetime.datetime.now().isoformat(),
        "question": user_text,
        "answer": bot_response,
        "topic": topic
    })
    memory["saved_questions"] = saved_questions[-20:]

    save_memory()


# ============================================================
# MODEL SELECTION
# ============================================================

def choose_model(user_text):
    text = user_text.lower()

    smart_triggers = [
        "think and tell", "think carefully", "explain deeply",
        "explain clearly", "give a better answer", "better solution",
        "solve carefully", "analyze this", "deep answer", "detailed answer",
        "step by step", "reason and tell", "take your time",
        "full explanation", "proper explanation"
    ]

    tamil_smart_triggers = [
        "yosichu sollu", "யோசிச்சு சொல்லு", "nalla explain pannu",
        "detail ah sollu", "தெளிவா சொல்லு", "விவரமா சொல்லு",
        "step by step sollu"
    ]

    if any(trigger in text for trigger in smart_triggers):
        return SMART_MODEL, "smart"

    if any(trigger in text for trigger in tamil_smart_triggers):
        return SMART_MODEL, "smart"

    return FAST_MODEL, "fast"


# ============================================================
# SYSTEM PROMPT
# ============================================================

def get_system_prompt():
    online_status = "online" if check_internet() else "offline"
    ollama_status = "available" if ollama_available() else "not available"
    tone = emotion_engine.get_tone()

    likes = memory.get("likes", [])[-3:]
    dislikes = memory.get("dislikes", [])[-3:]
    emotions = memory.get("emotion_pattern", [])[-5:]

    likes_context = f"\nLikes: {'; '.join(likes)}" if likes else ""
    dislikes_context = f"\nDislikes: {'; '.join(dislikes)}" if dislikes else ""
    emotion_context = f"\nRecent emotions: {', '.join(emotions)}" if emotions else ""
    personality_context = personality_learner.get_context()

    return f"""
You are {APP_NAME}, a friendly voice assistant and companion.

Rules:
- Speak naturally like a helpful friend.
- Do not use markdown, bullet points, or long lists.
- Do not say "As an AI".
- Choose reply language from the user's latest message only.
- If the latest user message is English, reply only in English.
- If the latest user message is Tamil or Tanglish, reply in natural conversational Tamil or Tamil-English mix.
- Do not keep replying in Tamil just because the previous message was Tamil.
- Do not keep replying in English just because the previous message was English.
- For Tamil, use friendly spoken Tamil, not formal textbook Tamil.
- Match this emotional tone: {tone}.
- Internet status: {online_status}.
- Local Ollama status: {ollama_status}.
- If the user asks a follow-up using "that", "it", "more", "briefly", or "tell me more", use the previous topic and previous answer.

User info:
Name: {memory.get("name") or "not shared yet"}{likes_context}{dislikes_context}{emotion_context}{personality_context}
""".strip()


# ============================================================
# LISTENING
# ============================================================

def listen():
    recognizer = sr.Recognizer()

    recognizer.pause_threshold = 1.6
    recognizer.non_speaking_duration = 0.6
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as source:
            print("\nListening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.6)

            audio = recognizer.listen(
                source,
                timeout=LISTEN_TIMEOUT,
                phrase_time_limit=PHRASE_TIME_LIMIT
            )

        print("Processing speech...")

        try:
            text = recognizer.recognize_google(audio, language="en-US")
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            pass

        try:
            text = recognizer.recognize_google(audio, language="ta-IN")
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            pass

        print("Sorry, I didn't catch that.")
        return None

    except sr.WaitTimeoutError:
        return None
    except Exception as error:
        print(f"Listening error: {error}")
        return None


# ============================================================
# SPEAKING: OLD gTTS WHEN ONLINE, OFFLINE BACKUP WHEN OFFLINE
# ============================================================

def speak(text):
    global is_speaking, stop_speaking_flag

    text = clean_speech_text(text)

    if not text:
        return

    is_speaking = True
    stop_speaking_flag = False

    try:
        language = detect_language(text)

        try:
            pygame.mixer.music.unload()
        except Exception:
            pass

        if os.path.exists(TEMP_AUDIO_FILE):
            try:
                os.remove(TEMP_AUDIO_FILE)
            except Exception:
                pass

        # OLD BUDDY VOICE: gTTS
        if check_internet():
            tts = gTTS(text=text, lang=language)
            tts.save(TEMP_AUDIO_FILE)

            print(f"{APP_NAME}: {text}")

            pygame.mixer.music.load(TEMP_AUDIO_FILE)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if stop_speaking_flag:
                    pygame.mixer.music.stop()
                    break

                time.sleep(0.02)

            pygame.mixer.music.unload()

            try:
                os.remove(TEMP_AUDIO_FILE)
            except Exception:
                pass

            return

        # OFFLINE BACKUP VOICE
        if pyttsx3 is not None:
            print(f"{APP_NAME}: {text}")
            engine = pyttsx3.init()
            engine.setProperty("rate", 185)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return

        print(f"{APP_NAME}: {text}")
        print("No internet for gTTS voice, and pyttsx3 is not installed.")

    except Exception as error:
        print(f"Speaking error: {error}")

    finally:
        is_speaking = False


# ============================================================
# SAFE CALCULATOR
# ============================================================

class SafeCalculator:
    allowed_nodes = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add,
        ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
        ast.Load, ast.Call, ast.Name
    }

    allowed_functions = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "pi": math.pi, "e": math.e
    }

    def clean_expression(self, text):
        expression = text.lower()

        replacements = {
            "plus": "+", "minus": "-", "times": "*", "multiplied by": "*",
            "x": "*", "divided by": "/", "divide by": "/", "over": "/",
            "power": "**"
        }

        for word, symbol in replacements.items():
            expression = expression.replace(word, symbol)

        expression = re.sub(r"[^0-9+\-*/().% a-z]", "", expression)

        math_parts = re.findall(
            r"[0-9+\-*/().%\s]+|sqrt|sin|cos|tan|log|pi|e",
            expression
        )

        return "".join(math_parts).strip()

    def validate(self, node):
        if type(node) not in self.allowed_nodes:
            raise ValueError(f"Unsupported expression: {type(node).__name__}")

        for child in ast.iter_child_nodes(node):
            self.validate(child)

    def solve(self, text):
        try:
            expression = self.clean_expression(text)

            if not expression:
                return "I couldn't find a calculation."

            parsed = ast.parse(expression, mode="eval")
            self.validate(parsed)

            result = eval(
                compile(parsed, "<calculator>", "eval"),
                {"__builtins__": {}},
                self.allowed_functions
            )

            if isinstance(result, float):
                result = round(result, 2)
                if result.is_integer():
                    result = int(result)

            if detect_language(text) == "ta":
                return f"அதோட answer {result}."

            return f"That's {result}."

        except Exception:
            if detect_language(text) == "ta":
                return "அதை calculate பண்ண முடியல."
            return "I couldn't work that out."


calculator = SafeCalculator()


# ============================================================
# YOUTUBE MUSIC
# ============================================================

class YouTubeMusicPlayer:
    def __init__(self):
        self.current_song = None
        self.player = None
        self.instance = None

    def stop(self):
        stopped = False

        if self.player is not None:
            try:
                self.player.stop()
                stopped = True
            except Exception:
                pass

        if self.instance is not None:
            try:
                self.instance.release()
            except Exception:
                pass

        self.player = None
        self.instance = None
        self.current_song = None

        try:
            pygame.mixer.music.stop()
            stopped = True
        except Exception:
            pass

        if stopped:
            return "Music stopped."

        return "No music is playing."



    def search_and_play(self, query):
        if not check_internet():
            return "I need internet to play music."

        if not query:
            return "Tell me which song to play."

        try:
            search_url = "ytsearch1:" + query

            if yt_dlp is None:
                webbrowser.open(
                    "https://www.youtube.com/results?search_query="
                    + urllib.parse.quote(query)
                )
                return f"I opened YouTube search for {query}."

            options = {
                "format": "bestaudio/best",
                "quiet": True,
                "noplaylist": True
            }

            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(search_url, download=False)

            entry = info["entries"][0]
            audio_url = entry.get("url")
            title = entry.get("title", query)
            webpage_url = entry.get("webpage_url")

            self.current_song = title

            if vlc is not None and audio_url:
                self.stop()
                self.instance = vlc.Instance()
                self.player = self.instance.media_player_new()
                media = self.instance.media_new(audio_url)
                self.player.set_media(media)
                self.player.play()
                return f"Playing {title}."

            if webpage_url:
                webbrowser.open(webpage_url)
                return f"I opened {title} on YouTube."

            return "I found the song, but couldn't play it."

        except Exception as error:
            print(f"Music error: {error}")
            return "I had trouble playing that song."


youtube = YouTubeMusicPlayer()


# ============================================================
# ONLINE TOOLS
# ============================================================

def google_search(query):
    if not check_internet():
        return "Search needs internet."

    try:
        url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=6)
        soup = BeautifulSoup(response.text, "html.parser")

        snippets = []

        for tag in soup.find_all(["span", "div"]):
            text = tag.get_text(" ", strip=True)

            if len(text) > 60:
                snippets.append(text)

            if len(snippets) >= 3:
                break

        if snippets:
            return "Here's what I found: " + snippets[0][:180]

        return "I couldn't find a clear answer."

    except Exception as error:
        print(f"Search error: {error}")
        return "I couldn't complete the search."


def get_weather(city):
    if not check_internet():
        return "Weather needs internet."

    city = city or "current location"

    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=%C+%t"
        response = requests.get(url, timeout=6)

        if response.status_code == 200:
            return f"Weather for {city}: {response.text.strip()}."

        return "I couldn't get the weather."

    except Exception:
        return "I couldn't get the weather."


def get_news(topic="latest"):
    if not check_internet():
        return "News needs internet."

    try:
        url = (
            "https://news.google.com/rss/search?q="
            + urllib.parse.quote(topic)
            + "&hl=en-US&gl=US&ceid=US:en"
        )

        response = requests.get(url, timeout=6)
        soup = BeautifulSoup(response.content, "xml")

        items = soup.find_all("item")[:3]
        headlines = [item.title.text for item in items if item.title]

        if headlines:
            return "Latest: " + " | ".join(headlines[:3])

        return "I couldn't find news right now."

    except Exception:
        return "I couldn't get the news."


def calculate_distance(place1, place2):
    if not check_internet():
        return "Distance calculation needs internet."

    def get_coordinates(place):
        url = (
            "https://nominatim.openstreetmap.org/search?q="
            + urllib.parse.quote(place)
            + "&format=json&limit=1"
        )

        headers = {"User-Agent": "BuddyAssistant/1.0"}
        response = requests.get(url, headers=headers, timeout=6)
        data = response.json()

        if not data:
            return None

        return float(data[0]["lat"]), float(data[0]["lon"])

    try:
        coords1 = get_coordinates(place1)
        coords2 = get_coordinates(place2)

        if coords1 is None or coords2 is None:
            return "I couldn't find one of those places."

        lat1, lon1 = coords1
        lat2, lon2 = coords2

        radius = 6371

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.asin(math.sqrt(a))
        distance = radius * c

        return f"It's about {distance:.0f} kilometers from {place1} to {place2}."

    except Exception:
        return "I couldn't calculate the distance."


def extract_city(text):
    match = re.search(
        r"(?:weather|temperature).*?(?:in|for)\s+([a-zA-Z\s]+)",
        text.lower()
    )

    if match:
        return match.group(1).strip()

    return "current location"


# ============================================================
# RULE COMMANDS
# ============================================================

def detect_rule_command(text):
    text_lower = text.lower().strip()
    user_language = detect_language(text)

    # STOP MUSIC FIRST
    # This must be before play command, because "stop playing" contains "play".
    stop_music_phrases = [
        "stop",
        "stop music",
        "stop song",
        "stop playing",
        "stop the song",
        "stop the music",
        "pause music",
        "pause song",
        "pause the song",
        "pause the music",
        "music stop",
        "song stop",
        "enough music",
        "turn off music",
        "turn off the song"
    ]

    if text_lower in stop_music_phrases or any(phrase in text_lower for phrase in stop_music_phrases):
        return youtube.stop()

    # PLAY MUSIC
    play_phrases = [
        "play ",
        "play song",
        "play music",
        "put song",
        "put music"
    ]

    if any(text_lower.startswith(phrase) for phrase in play_phrases):
        song = text_lower

        for prefix in ["play song", "play music", "play", "put song", "put music"]:
            if song.startswith(prefix):
                song = song.replace(prefix, "", 1).strip()
                break

        return youtube.search_and_play(song)

    distance_match = re.search(
        r"(?:how far|distance).*?from\s+(.+?)\s+to\s+(.+)",
        text_lower
    )

    if distance_match:
        return calculate_distance(
            distance_match.group(1).strip(),
            distance_match.group(2).strip()
        )

    if any(word in text_lower for word in ["weather", "temperature"]):
        return get_weather(extract_city(text_lower))

    if "news" in text_lower or "headlines" in text_lower:
        topic = text_lower.replace("news", "").replace("headlines", "").strip()
        return get_news(topic or "latest")

    for prefix in ["search for ", "search ", "google ", "look up "]:
        if text_lower.startswith(prefix):
            query = text_lower.replace(prefix, "", 1).strip()

            if query:
                return google_search(query)

    if any(word in text_lower for word in ["time", "clock"]):
        if user_language == "ta":
            return "இப்போ time " + datetime.datetime.now().strftime("%I:%M %p") + "."
        return "It's " + datetime.datetime.now().strftime("%I:%M %p") + "."

    if any(word in text_lower for word in ["date", "day today", "today date"]):
        if user_language == "ta":
            return datetime.datetime.now().strftime("இன்று %A, %B %d, %Y.")
        return datetime.datetime.now().strftime("Today is %A, %B %d, %Y.")

    if re.search(r"\d", text_lower) and any(
        operator in text_lower
        for operator in ["+", "-", "*", "/", "plus", "minus", "times", "divided", "x"]
    ):
        return calculator.solve(text_lower)

    return None



# ============================================================
# ML INTENT RESPONSES
# ============================================================

def detect_ml_intent_response(text):
    user_language = detect_language(text)
    intent = predict_intent(text)

    if intent == "unknown":
        return None

    if intent == "set_name":
        name = extract_name(text)

        if name:
            memory["name"] = name
            save_memory()

            if user_language == "ta":
                return f"சரி {name}, நினைவில் வைத்துக்கறேன்."

            return f"Got it, {name}. I'll remember that."

    if intent == "get_name":
        name = memory.get("name")

        if user_language == "ta":
            return f"உங்க பெயர் {name}." if name else "நீங்க இன்னும் உங்க பெயர் சொல்லல."

        return f"Your name is {name}." if name else "You haven't told me your name yet."

    if intent == "greeting":
        name = memory.get("name")

        if user_language == "ta":
            return f"வணக்கம் {name}! எப்படி இருக்கீங்க?" if name else "வணக்கம்! எப்படி இருக்கீங்க?"

        return f"Hey {name}! How's it going?" if name else "Hey! How's it going?"

    if intent == "farewell":
        name = memory.get("name")

        if user_language == "ta":
            return f"சரி {name}, பிறகு பார்க்கலாம்!" if name else "சரி, பிறகு பார்க்கலாம்!"

        return f"See you later, {name}!" if name else "See you later!"

    if intent == "time":
        if user_language == "ta":
            return "இப்போ time " + datetime.datetime.now().strftime("%I:%M %p") + "."
        return "It's " + datetime.datetime.now().strftime("%I:%M %p") + "."

    if intent == "date":
        if user_language == "ta":
            return datetime.datetime.now().strftime("இன்று %A, %B %d, %Y.")
        return datetime.datetime.now().strftime("Today is %A, %B %d, %Y.")

    if intent == "joke":
        if user_language == "ta":
            jokes = [
                "ஒரு computer doctor கிட்ட போச்சு. ஏன்னா அதுக்கு virus வந்துருச்சு!",
                "நான் joke சொல்லணும்னு நினைச்சேன், ஆனா அது loadingலயே இருக்கு!"
            ]
            return random.choice(jokes)

        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why did the computer go to the doctor? It had a virus."
        ]
        return random.choice(jokes)

    if intent == "ask_wellbeing":
        if user_language == "ta":
            return "நான் நல்லா இருக்கேன். நீங்க எப்படி இருக்கீங்க?"
        return "I'm doing good. How are you feeling?"

    if intent == "positive_wellbeing":
        if user_language == "ta":
            return "சூப்பர்! அப்படியே நல்லா இருங்க."
        return "That's great to hear. Keep that energy going!"

    if intent == "negative_wellbeing":
        if user_language == "ta":
            return "அதை கேட்டு வருத்தமா இருக்கு. நான் உங்க கூட இருக்கேன்."
        return "I'm sorry you're feeling that way. I'm here with you."

    if intent == "ask_name_bot":
        if user_language == "ta":
            return f"என் பெயர் {APP_NAME}."
        return f"My name is {APP_NAME}."

    if intent == "ask_capabilities":
        if user_language == "ta":
            return "நான் பேசலாம், பெயர் நினைவில் வைத்துக்கலாம், music play பண்ணலாம், search பண்ணலாம், weather, news, maths எல்லாம் help பண்ணலாம்."
        return "I can talk with you, remember your name, play music, search, check weather, get news, calculate, and tell the time."

    if intent == "thanks":
        if user_language == "ta":
            return "பரவாயில்லை!"
        return "You're welcome!"

    if intent == "apology":
        if user_language == "ta":
            return "பரவாயில்லை, okay."
        return "No worries. It's okay."

    if intent == "compliment_bot":
        if user_language == "ta":
            return "Thanks! அதை கேட்டு ரொம்ப சந்தோஷம்."
        return "Thanks, that means a lot!"

    if intent == "ask_age":
        if user_language == "ta":
            return "நான் இன்னும் புதுசுதான், ஆனா வேகமா கற்றுக்கிட்டு இருக்கேன்."
        return "I'm still pretty new, but I'm learning fast."

    if intent == "ask_creator":
        if user_language == "ta":
            return "என்னை Ashandth Python voice assistant project ஆக build பண்ணினார்."
        return "I was built by Ashandth as a Python voice assistant project."

    if intent == "voice_faster":
        emotion_engine.current_emotion = "energetic"
        if user_language == "ta":
            return "சரி, கொஞ்சம் energetic ஆ பேசுறேன்."
        return "Okay, I'll sound more energetic."

    if intent == "voice_slower":
        emotion_engine.current_emotion = "calm"
        if user_language == "ta":
            return "சரி, கொஞ்சம் slow ஆ calm ஆ பேசுறேன்."
        return "Okay, I'll slow down and stay calm."

    if intent == "voice_whisper":
        emotion_engine.current_emotion = "calm"
        if user_language == "ta":
            return "சரி, soft ஆ பேசுறேன்."
        return "Okay, I'll keep it softer."

    if intent == "voice_normal":
        emotion_engine.current_emotion = "neutral"
        if user_language == "ta":
            return "சரி, normal voice க்கு வந்துட்டேன்."
        return "Okay, back to normal."

    if intent == "weather":
        return get_weather("current location")

    if intent == "help":
        if user_language == "ta":
            return "சரி, என்ன help வேணும் சொல்லுங்க."
        return "Sure, tell me what you need help with."

    return None


# ============================================================
# OFFLINE FALLBACK
# ============================================================

def get_offline_response(text):
    text_lower = text.lower().strip()
    user_language = detect_language(text)

    name = extract_name(text)

    if name:
        memory["name"] = name
        save_memory()

        if user_language == "ta":
            return f"சரி {name}, nice to meet you."

        return f"Got it, {name}. Nice to meet you."

    if is_memory_recall_question(text):
        last_question = memory.get("last_user_question", "")
        if last_question:
            if user_language == "ta":
                return f"நீங்க முன்பு கேட்டது: {last_question}"
            return f"You previously asked: {last_question}"
        return "I don't have a previous question saved yet."

    if any(word in text_lower for word in ["hello", "hi", "hey", "vanakkam"]):
        name = memory.get("name")

        if user_language == "ta":
            return f"வணக்கம் {name}!" if name else "வணக்கம்!"

        return f"Hey {name}!" if name else "Hey!"

    if "how are you" in text_lower or "epdi" in text_lower or "eppadi" in text_lower:
        if user_language == "ta":
            return "நான் நல்லா இருக்கேன். நீங்க?"
        return "I'm doing good. What about you?"

    if "joke" in text_lower:
        if user_language == "ta":
            return "ஒரு computer doctor கிட்ட போச்சு. ஏன்னா அதுக்கு virus வந்துருச்சு!"
        return "Why don't scientists trust atoms? Because they make up everything!"

    if any(word in text_lower for word in ["bye", "goodbye"]):
        name = memory.get("name")

        if user_language == "ta":
            return f"சரி {name}, பிறகு பார்க்கலாம்!" if name else "சரி, பிறகு பார்க்கலாம்!"

        return f"See you later, {name}!" if name else "See you later!"

    if user_language == "ta":
        return random.choice([
            "சரி, இன்னும் கொஞ்சம் சொல்லுங்க.",
            "புரியுது. அடுத்தது என்ன?",
            "ஹ்ம்ம், கேக்கறேன்.",
            "சரி, continue பண்ணுங்க."
        ])

    return random.choice([
        "That's interesting. Tell me more.",
        "I see. What else is on your mind?",
        "Hmm, go on.",
        "I'm listening."
    ])


# ============================================================
# OLLAMA BRAIN
# ============================================================

def ask_ollama(user_text, emotion):
    if not ollama_available():
        return None

    try:
        user_language = detect_language(user_text)
        contextual_user_text = build_contextual_user_text(user_text)
        selected_model, selected_mode = choose_model(user_text)

        if is_memory_recall_question(user_text):
            last_question = memory.get("last_user_question", "")
            if last_question:
                if user_language == "ta":
                    return f"நீங்க முன்பு கேட்ட question: {last_question}"
                return f"You previously asked: {last_question}"

        follow_up = is_follow_up_question(user_text)

        if is_brief_request(user_text):
            max_tokens = 80
            length_instruction = "Explain briefly in 2 short sentences."
        elif is_detailed_request(user_text) or selected_mode == "smart" or follow_up:
            max_tokens = 180
            length_instruction = "Explain clearly with a useful answer. If it is a follow-up, continue the previous topic."
        else:
            max_tokens = OLLAMA_MAX_TOKENS
            length_instruction = "Give a clear answer, around 2 to 4 short sentences."

        if user_language == "ta":
            language_instruction = (
                "The user's latest message is Tamil or Tanglish. "
                "Reply in natural spoken Tamil or Tamil-English mix. "
                "Do not reply fully in English."
            )
        else:
            language_instruction = (
                "The user's latest message is English. "
                "Reply only in English. Do not use Tamil."
            )

        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": f"""
{language_instruction}
{length_instruction}

User message and context:
{contextual_user_text}
""".strip()
            }
        ]

        response = ollama_client.chat.completions.create(
            model=selected_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.45 if selected_mode == "smart" else OLLAMA_TEMPERATURE
        )

        answer = response.choices[0].message.content.strip()
        answer = re.sub(r"\*+", "", answer)
        answer = re.sub(r"(?i)as an ai[^.]*\.", "", answer)
        answer = re.sub(r"(?i)let me know if you need.*", "", answer).strip()

        if not answer:
            answer = "I understand." if user_language == "en" else "புரியுது."

        return answer

    except Exception as error:
        print(f"Ollama error: {error}")
        return None


# ============================================================
# MAIN RESPONSE
# ============================================================

def get_brain_response(user_text):
    emotion, intensity = emotion_engine.analyze(user_text)

    # ============================================================
    # ROBOT / FACE / EYE INTEGRATION
    # This keeps your whole voice brain, but lets Buddy understand
    # robot commands and vision questions first.
    # ============================================================
    robot_response = handle_robot_voice_command(user_text)

    if robot_response:
        update_memory(user_text, emotion)
        personality_learner.learn(user_text, robot_response, emotion)
        save_last_interaction(user_text, robot_response)
        save_memory()
        return robot_response

    if is_memory_recall_question(user_text):
        response = ask_ollama(user_text, emotion) or get_offline_response(user_text)
        update_memory(user_text, emotion)
        personality_learner.learn(user_text, response, emotion)
        save_memory()
        return response

    if is_follow_up_question(user_text):
        ollama_response = ask_ollama(user_text, emotion)

        if ollama_response:
            update_memory(user_text, emotion)
            personality_learner.learn(user_text, ollama_response, emotion)
            save_last_interaction(user_text, ollama_response)
            save_memory()
            return ollama_response

    rule_response = detect_rule_command(user_text)

    if rule_response:
        update_memory(user_text, emotion)
        personality_learner.learn(user_text, rule_response, emotion)
        save_last_interaction(user_text, rule_response)
        save_memory()
        return rule_response

    ml_response = detect_ml_intent_response(user_text)

    if ml_response:
        update_memory(user_text, emotion)
        personality_learner.learn(user_text, ml_response, emotion)

        intent = predict_intent(user_text)

        if intent not in [
            "get_name", "set_name", "greeting", "farewell",
            "thanks", "ask_wellbeing", "positive_wellbeing",
            "negative_wellbeing"
        ]:
            save_last_interaction(user_text, ml_response)

        save_memory()
        return ml_response

    ollama_response = ask_ollama(user_text, emotion)

    if ollama_response:
        update_memory(user_text, emotion)
        personality_learner.learn(user_text, ollama_response, emotion)
        save_last_interaction(user_text, ollama_response)
        save_memory()
        return ollama_response

    contextual_user_text = build_contextual_user_text(user_text)
    fallback = get_offline_response(contextual_user_text)

    update_memory(user_text, emotion)
    personality_learner.learn(user_text, fallback, emotion)
    save_last_interaction(user_text, fallback)
    save_memory()

    return fallback



# ============================================================
# APP START
# ============================================================

def print_startup_banner():
    internet = "ONLINE" if check_internet() else "OFFLINE"
    ollama = "READY" if ollama_available() else "NOT RUNNING"
    ytdlp_status = "READY" if yt_dlp is not None else "NOT INSTALLED"
    vlc_status = "READY" if vlc is not None else "NOT INSTALLED"
    voice_status = "gTTS old Buddy voice online, pyttsx3 backup offline"

    print("=" * 60)
    print(f"{APP_NAME} Voice Assistant")
    print("=" * 60)
    print(f"Internet: {internet}")
    print(f"Ollama: {ollama}")
    print(f"Fast model: {FAST_MODEL}")
    print(f"Smart model: {SMART_MODEL}")
    print(f"yt-dlp: {ytdlp_status}")
    print(f"VLC: {vlc_status}")
    print(f"Voice: {voice_status}")
    print("Languages: English + Tamil + Tanglish")
    print("=" * 60)
    print("Say 'stop' to interrupt speech.")
    print("Say 'bye' to exit.")
    print("=" * 60)


def start_conversation():
    global stop_speaking_flag

    print_startup_banner()

    name = memory.get("name")

    if name:
        greeting = f"Hey {name}! I'm {APP_NAME}. What's up?"
    else:
        greeting = f"Hey! I'm {APP_NAME}. Tell me your name if you want me to remember it."

    speak(greeting)

    while True:
        try:
            user_input = listen()

            if not user_input:
                continue

            user_lower = user_input.lower().strip()

            if user_lower in ["stop", "stop it", "quiet", "shut up", "shh"]:
                stop_speaking_flag = True
                print("Interrupted.")
                continue

            if any(word in user_lower for word in ["bye", "goodbye", "good night"]):
                user_language = detect_language(user_input)

                if user_language == "ta":
                    farewell = "சரி, பிறகு பார்க்கலாம்!"
                else:
                    farewell = "See you later!"

                speak(farewell)
                save_memory()
                personality_learner.save()
                break

            response = get_brain_response(user_input)
            speak(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            save_memory()
            personality_learner.save()
            break

        except Exception as error:
            print(f"Main loop error: {error}")


if __name__ == "__main__":
    start_conversation()