import ast
import datetime
import json
import math
import os
import queue
import random
import re
import threading
import time
import urllib.parse
import webbrowser
from collections import deque

import pygame
import requests
import speech_recognition as sr
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
MAX_HISTORY = 12

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:latest"

TEMP_AUDIO_FILE = "temp_response.mp3"
FAST_VOICE = True

pygame.mixer.init()

is_speaking = False
stop_speaking_flag = False


def check_internet():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False


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
            json.dump(data, file, indent=2)
    except Exception as error:
        print(f"Could not save {path}: {error}")


def load_memory():
    default = {
        "name": None,
        "conversation_count": 0,
        "likes": [],
        "dislikes": [],
        "recent_topics": [],
        "emotion_pattern": []
    }

    data = safe_json_load(MEMORY_FILE, default)

    for key, value in default.items():
        data.setdefault(key, value)

    return data


def save_memory():
    safe_json_save(MEMORY_FILE, memory)


memory = load_memory()


class EmotionEngine:
    def __init__(self):
        self.current_emotion = "neutral"
        self.emotion_history = deque(maxlen=20)
        self.emotion_intensity = 0.5

        self.emotion_map = {
            "joy": [
                "happy", "great", "amazing", "wonderful", "love", "excited",
                "fantastic", "awesome", "thrilled", "good"
            ],
            "sadness": [
                "sad", "upset", "unhappy", "depressed", "miserable",
                "heartbroken", "crying", "lonely"
            ],
            "anger": [
                "angry", "frustrated", "furious", "mad", "annoyed", "irritated"
            ],
            "fear": [
                "scared", "afraid", "terrified", "anxious", "worried", "nervous"
            ],
            "surprise": [
                "wow", "omg", "unbelievable", "shocking", "surprised", "incredible"
            ],
            "calm": [
                "peaceful", "relaxed", "chill", "tired", "sleepy", "quiet"
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
            "surprise": "curious and engaged",
            "calm": "relaxed and peaceful",
            "energetic": "enthusiastic and lively",
            "neutral": "natural and conversational"
        }

        return tones.get(self.current_emotion, "natural and conversational")


emotion_engine = EmotionEngine()


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
            "music": ["music", "song", "band", "concert"],
            "movies": ["movie", "film", "cinema", "netflix"],
            "food": ["food", "cooking", "restaurant", "recipe"],
            "sports": ["sports", "game", "team", "player"],
            "travel": ["travel", "trip", "vacation"],
            "work": ["work", "job", "career", "office"],
            "family": ["family", "mom", "dad", "brother", "sister"],
            "study": ["exam", "study", "school", "college", "maths", "assignment"]
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
        return f"\nUser's favorite topics: {topic_names}"

    def save(self):
        safe_json_save(PERSONALITY_FILE, self.data)


personality_learner = PersonalityLearner()


def create_ollama_client():
    try:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"
        )
        return client
    except Exception as error:
        print(f"Ollama client failed: {error}")
        return None


ollama_client = create_ollama_client()


def ollama_available():
    if ollama_client is None:
        return False

    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def extract_name(text):
    patterns = [
        r"\bmy name is\s+(.+)",
        r"\bcall me\s+(.+)",
        r"\bi am\s+(.+)",
        r"\bi'm\s+(.+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower().strip())

        if match:
            name = match.group(1).strip()
            name = re.sub(r"[^a-zA-Z\s]", "", name).strip()

            if name:
                return name.title()

    return None


def update_memory(user_text, emotion):
    text = user_text.lower()

    memory["conversation_count"] = memory.get("conversation_count", 0) + 1

    if any(word in text for word in ["love", "like", "enjoy", "favorite"]):
        memory["likes"].append(user_text)
        memory["likes"] = memory["likes"][-20:]

    if any(word in text for word in ["hate", "dislike", "can't stand", "boring"]):
        memory["dislikes"].append(user_text)
        memory["dislikes"] = memory["dislikes"][-20:]

    memory["recent_topics"].append(text)
    memory["recent_topics"] = memory["recent_topics"][-10:]

    memory["emotion_pattern"].append(emotion)
    memory["emotion_pattern"] = memory["emotion_pattern"][-50:]


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
- Keep replies short, usually 1 to 3 sentences.
- Do not use markdown, bullet points, or long lists.
- Do not say "As an AI".
- Match this emotional tone: {tone}.
- Internet status: {online_status}.
- Local Ollama status: {ollama_status}.

User info:
Name: {memory.get("name") or "not shared yet"}{likes_context}{dislikes_context}{emotion_context}{personality_context}
""".strip()


conversation_history = [
    {"role": "system", "content": get_system_prompt()}
]


def listen():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("\nListening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)

        print("Processing speech...")
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text

    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
        return None
    except Exception as error:
        print(f"Listening error: {error}")
        return None


def clean_speech_text(text):
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def speak(text):
    global is_speaking, stop_speaking_flag

    text = clean_speech_text(text)

    if not text:
        return

    print(f"{APP_NAME}: {text}")

    is_speaking = True
    stop_speaking_flag = False

    try:
        # Previous Buddy voice: gTTS
        if os.path.exists(TEMP_AUDIO_FILE):
            try:
                os.remove(TEMP_AUDIO_FILE)
            except Exception:
                pass

        tts = gTTS(text=text, lang="en")
        tts.save(TEMP_AUDIO_FILE)

        pygame.mixer.music.load(TEMP_AUDIO_FILE)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            if stop_speaking_flag:
                pygame.mixer.music.stop()
                break

            time.sleep(0.05)

        pygame.mixer.music.unload()

        try:
            os.remove(TEMP_AUDIO_FILE)
        except Exception:
            pass

    except Exception as error:
        print(f"Speaking error: {error}")

    finally:
        is_speaking = False

class SafeCalculator:
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Call,
        ast.Name
    }

    allowed_functions = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "pi": math.pi,
        "e": math.e
    }

    def clean_expression(self, text):
        expression = text.lower()

        replacements = {
            "plus": "+",
            "minus": "-",
            "times": "*",
            "multiplied by": "*",
            "x": "*",
            "divided by": "/",
            "divide by": "/",
            "over": "/",
            "power": "**"
        }

        for word, symbol in replacements.items():
            expression = expression.replace(word, symbol)

        expression = re.sub(r"[^0-9+\-*/().% a-z]", "", expression)

        math_parts = re.findall(r"[0-9+\-*/().%\s]+|sqrt|sin|cos|tan|log|pi|e", expression)
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

            return f"That's {result}."

        except Exception:
            return "I couldn't work that out."


calculator = SafeCalculator()


class YouTubeMusicPlayer:
    def __init__(self):
        self.current_song = None
        self.player = None
        self.instance = None

    def stop(self):
        if self.player is not None:
            try:
                self.player.stop()
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

        return "Music stopped."

    def search_and_play(self, query):
        if not check_internet():
            return "I need internet to play music."

        if not query:
            return "Tell me which song to play."

        try:
            search_url = "ytsearch1:" + query

            if yt_dlp is None:
                webbrowser.open("https://www.youtube.com/results?search_query=" + urllib.parse.quote(query))
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


def google_search(query):
    if not check_internet():
        return "Search needs internet."

    try:
        url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(response.text, "html.parser")

        snippets = []

        for tag in soup.find_all(["span", "div"]):
            text = tag.get_text(" ", strip=True)

            if len(text) > 60 and query.lower().split()[0] in text.lower():
                snippets.append(text)

            if len(snippets) >= 3:
                break

        if snippets:
            return "Here's what I found: " + snippets[0][:220]

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
        response = requests.get(url, timeout=8)

        if response.status_code == 200:
            return f"Weather for {city}: {response.text.strip()}."

        return "I couldn't get the weather."

    except Exception:
        return "I couldn't get the weather."


def get_news(topic="latest"):
    if not check_internet():
        return "News needs internet."

    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(topic)}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=8)
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
        response = requests.get(url, headers=headers, timeout=8)
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
    match = re.search(r"(?:weather|temperature).*?(?:in|for)\s+([a-zA-Z\s]+)", text.lower())

    if match:
        return match.group(1).strip()

    return "current location"


def detect_rule_command(text):
    text_lower = text.lower().strip()

    if text_lower in ["stop music", "stop song", "stop playing"]:
        return youtube.stop()

    if any(word in text_lower for word in ["play ", "play song", "play music"]):
        song = text_lower

        for prefix in ["play song", "play music", "play"]:
            if song.startswith(prefix):
                song = song.replace(prefix, "", 1).strip()
                break

        return youtube.search_and_play(song)

    distance_match = re.search(r"(?:how far|distance).*?from\s+(.+?)\s+to\s+(.+)", text_lower)

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
        return "It's " + datetime.datetime.now().strftime("%I:%M %p") + "."

    if any(word in text_lower for word in ["date", "day today", "today date"]):
        return datetime.datetime.now().strftime("Today is %A, %B %d, %Y.")

    if re.search(r"\d", text_lower) and any(
        operator in text_lower
        for operator in ["+", "-", "*", "/", "plus", "minus", "times", "divided", "x"]
    ):
        return calculator.solve(text_lower)

    return None


def detect_ml_intent_response(text):
    intent = predict_intent(text)

    if intent == "unknown":
        return None

    text_lower = text.lower()

    if intent == "set_name":
        name = extract_name(text)

        if name:
            memory["name"] = name
            save_memory()
            return f"Got it, {name}. I'll remember that."

    if intent == "get_name":
        name = memory.get("name")
        return f"Your name is {name}." if name else "You haven't told me your name yet."

    if intent == "greeting":
        name = memory.get("name")
        return f"Hey {name}! How's it going?" if name else "Hey! How's it going?"

    if intent == "farewell":
        name = memory.get("name")
        return f"See you later, {name}!" if name else "See you later!"

    if intent == "time":
        return "It's " + datetime.datetime.now().strftime("%I:%M %p") + "."

    if intent == "date":
        return datetime.datetime.now().strftime("Today is %A, %B %d, %Y.")

    if intent == "joke":
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why did the computer go to the doctor? It had a virus."
        ]
        return random.choice(jokes)

    if intent == "ask_wellbeing":
        return "I'm doing good. How are you feeling?"

    if intent == "positive_wellbeing":
        return "That's great to hear. Keep that energy going!"

    if intent == "negative_wellbeing":
        return "I'm sorry you're feeling that way. I'm here with you."

    if intent == "ask_name_bot":
        return f"My name is {APP_NAME}."

    if intent == "ask_capabilities":
        return "I can talk with you, remember your name, answer questions, play music, search, check weather, get news, calculate, and tell the time."

    if intent == "thanks":
        return "You're welcome!"

    if intent == "apology":
        return "No worries. It's okay."

    if intent == "compliment_bot":
        return "Thanks, that means a lot!"

    if intent == "ask_age":
        return "I'm still pretty new, but I'm learning fast."

    if intent == "ask_creator":
        return "I was built by Ashandth as a Python voice assistant project."

    if intent == "voice_faster":
        emotion_engine.current_emotion = "energetic"
        return "Okay, I'll sound more energetic."

    if intent == "voice_slower":
        emotion_engine.current_emotion = "calm"
        return "Okay, I'll slow down and stay calm."

    if intent == "voice_whisper":
        emotion_engine.current_emotion = "calm"
        return "Okay, I'll keep it softer."

    if intent == "voice_normal":
        emotion_engine.current_emotion = "neutral"
        return "Okay, back to normal."

    if intent == "weather":
        return get_weather("current location")

    if intent == "help":
        return "Sure, tell me what you need help with."

    return None


def get_offline_response(text):
    text_lower = text.lower().strip()

    name = extract_name(text)

    if name:
        memory["name"] = name
        save_memory()
        return f"Got it, {name}. Nice to meet you."

    if any(word in text_lower for word in ["hello", "hi", "hey"]):
        name = memory.get("name")
        return f"Hey {name}!" if name else "Hey!"

    if "how are you" in text_lower:
        return "I'm doing good. What about you?"

    if "joke" in text_lower:
        return "Why don't scientists trust atoms? Because they make up everything!"

    if any(word in text_lower for word in ["bye", "goodbye"]):
        name = memory.get("name")
        return f"See you later, {name}!" if name else "See you later!"

    return random.choice([
        "That's interesting. Tell me more.",
        "I see. What else is on your mind?",
        "Hmm, go on.",
        "I'm listening."
    ])


def ask_ollama(user_text, emotion):
    global conversation_history

    if not ollama_available():
        return None

    try:
        conversation_history[0] = {
            "role": "system",
            "content": get_system_prompt()
        }

        if len(conversation_history) > MAX_HISTORY + 1:
            conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY:]

        tagged_input = f"[User emotion: {emotion}] {user_text}"
        conversation_history.append({"role": "user", "content": tagged_input})

        response = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=conversation_history,
            max_tokens=50,
            temperature=0.5
        )

        answer = response.choices[0].message.content.strip()
        answer = re.sub(r"\*+", "", answer)
        answer = re.sub(r"(?i)as an ai[^.]*\.", "", answer)
        answer = re.sub(r"(?i)let me know if you need.*", "", answer).strip()

        if not answer:
            answer = "I understand."

        conversation_history.append({"role": "assistant", "content": answer})
        return answer

    except Exception as error:
        print(f"Ollama error: {error}")
        return None


def get_brain_response(user_text):
    emotion, intensity = emotion_engine.analyze(user_text)

    rule_response = detect_rule_command(user_text)

    if rule_response:
        update_memory(user_text, emotion)
        personality_learner.learn(user_text, rule_response, emotion)
        save_memory()
        return rule_response

    ml_response = detect_ml_intent_response(user_text)

    if ml_response:
        update_memory(user_text, emotion)
        personality_learner.learn(user_text, ml_response, emotion)
        save_memory()
        return ml_response

    ollama_response = ask_ollama(user_text, emotion)

    if ollama_response:
        update_memory(user_text, emotion)
        personality_learner.learn(user_text, ollama_response, emotion)
        save_memory()
        return ollama_response

    fallback = get_offline_response(user_text)

    update_memory(user_text, emotion)
    personality_learner.learn(user_text, fallback, emotion)
    save_memory()

    return fallback


def print_startup_banner():
    internet = "ONLINE" if check_internet() else "OFFLINE"
    ollama = "READY" if ollama_available() else "NOT RUNNING"
    pyttsx3_status = "READY" if pyttsx3 is not None else "NOT INSTALLED"
    ytdlp_status = "READY" if yt_dlp is not None else "NOT INSTALLED"
    vlc_status = "READY" if vlc is not None else "NOT INSTALLED"

    print("=" * 60)
    print(f"{APP_NAME} Voice Assistant")
    print("=" * 60)
    print(f"Internet: {internet}")
    print(f"Ollama: {ollama}")
    print(f"pyttsx3: {pyttsx3_status}")
    print(f"yt-dlp: {ytdlp_status}")
    print(f"VLC: {vlc_status}")
    print("=" * 60)
    print("Say 'stop' to interrupt speech.")
    print("Say 'bye' to exit.")
    print("=" * 60)


def start_conversation():
    global stop_speaking_flag

    print_startup_banner()

    name = memory.get("name")
    greeting = f"Hey {name}! I'm {APP_NAME}. What's up?" if name else f"Hey! I'm {APP_NAME}. What's up?"

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