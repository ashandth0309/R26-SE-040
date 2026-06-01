import sounddevice as sd
import json
import random
from vosk import Model, KaldiRecognizer
import datetime
import time as tm
import pyttsx3
import threading
import queue
import sys
import re
from ml_brain import predict_intent

# ==========================================
# NAME EXTRACTION HELPER
# Set 5 Wednesday — regex-based name extraction for accuracy
# ==========================================
def extract_name(text):
    """
    Extracts user name from phrases like 'my name is ashan kumar' or 'call me ashan'.
    Uses regex so full names are captured correctly instead of just the last word.
    """
    match = re.search(r"(my name is|call me)\s+(.+)", text)
    if match:
        return match.group(2).strip().title()
    return text.split()[-1].capitalize()

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r"C:\Users\sobiy\Desktop\Robot - Copy - Copy\R26-SE-040\faceRecAndEmotion\models\vosk-model"

# Wake word configuration
WAKE_WORD = "buddy"
WAKE_WORD_TIMEOUT = 15  # seconds before going back to sleep after activation

# Voice modulation profiles (for voice changer feature - Set 2 Sunday plan)
VOICE_PROFILES = {
    "default": {"rate": 170, "volume": 1.0, "voice_index": 1},
    "energetic": {"rate": 200, "volume": 1.0, "voice_index": 1},
    "calm":     {"rate": 145, "volume": 0.9, "voice_index": 1},
    "whisper":  {"rate": 155, "volume": 0.6, "voice_index": 1},
}
current_voice_profile = "default"

# Confidence threshold - ignore low-confidence STT results (Set 3 Saturday)
# ML confidence threshold added Set 4 Sunday / Set 5 Tuesday (in ml_brain.py)
MIN_TEXT_LENGTH = 3          # ignore noise/single phonemes
CONFIDENCE_WORD_THRESHOLD = 2  # minimum number of words to process

# Initialize Vosk model
try:
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)   # Enable word-level confidence scores
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if the path to your Vosk model is correct.")
    sys.exit(1)


import os

# Fix: always save/load memory.json in the SAME folder as speech.py
# Prevents the bug where memory saves to a different directory depending
# on where Python is launched from (Set 5 Wednesday fix)
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

# ==========================================
# MEMORY SYSTEM
# Set 4 Friday — persistent memory using memory.json
# Allows assistant to remember user name and preferences
# across sessions for personalized responses
# Set 5 Wednesday — fixed: use absolute path + global keyword + error handling
# ==========================================
def load_memory():
    """
    Loads user memory from disk. Returns empty dict if file does not exist yet
    (first run) or if the file is corrupted.
    """
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print("Warning: memory.json is corrupted. Starting with empty memory.")
        return {}

def save_memory(data):
    """
    Saves memory dict to disk. Uses indent=2 for human-readable output.
    Prints confirmation so you can verify it is working.
    """
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Memory] Saved to {MEMORY_FILE}: {data}")
    except Exception as e:
        print(f"[Memory] ERROR - could not save: {e}")

memory = load_memory()



# ==========================================
# GLOBAL STATE
# ==========================================
conversation_history = []
user_name = None
chatbot_name = "Usha"
is_speaking = False
is_awake = False               # Wake word state flag
wake_timer = None              # Timer to go back to sleep
audio_queue = queue.Queue()

# ==========================================
# WAKE WORD ENGINE (Set 3 Wednesday design)
# ==========================================
def activate_wake():
    """Put the assistant into active listening mode."""
    global is_awake, wake_timer
    is_awake = True
    print(f"\n🟢 [{chatbot_name} ACTIVATED] — Listening for your command...")

    # Cancel existing timer if re-activated
    if wake_timer and wake_timer.is_alive():
        wake_timer.cancel()

    # Auto-sleep after timeout
    wake_timer = threading.Timer(WAKE_WORD_TIMEOUT, deactivate_wake)
    wake_timer.daemon = True
    wake_timer.start()

def deactivate_wake():
    """Return to passive/sleep mode."""
    global is_awake
    is_awake = False
    print(f"\n😴 [{chatbot_name}] Going back to sleep. Say '{WAKE_WORD}' to wake me.")

def reset_wake_timer():
    """Reset inactivity timer after each spoken response."""
    global wake_timer
    if wake_timer and wake_timer.is_alive():
        wake_timer.cancel()
    wake_timer = threading.Timer(WAKE_WORD_TIMEOUT, deactivate_wake)
    wake_timer.daemon = True
    wake_timer.start()

# ==========================================
# CONFIDENCE THRESHOLDING (Set 3 Saturday)
# ==========================================
def passes_confidence_check(text):
    """
    Basic confidence check using text heuristics.
    Filters noise, single phonemes, and incomplete fragments.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False
    words = text.strip().split()
    if len(words) < CONFIDENCE_WORD_THRESHOLD:
        # Allow single-word commands if they're meaningful keywords
        single_word_ok = ["yes", "no", "bye", "hello", "hi", "hey",
                          WAKE_WORD, "stop", "help", "thanks"]
        if words[0].lower() not in single_word_ok:
            return False
    # Filter out pure phonetic noise patterns (hm, uh, ah, mm, etc.)
    noise_pattern = re.compile(r"^[aeiou hm]+$", re.IGNORECASE)
    if noise_pattern.match(text.strip()):
        return False
    return True

# ==========================================
# TEXT TO SPEECH — with Voice Modulation (Set 2 Sunday / Set 3 Saturday)
# ==========================================
def speak(text, profile=None):
    """
    Thread-safe TTS with voice modulation profiles.
    profile can be: 'default', 'energetic', 'calm', 'whisper'
    """
    global is_speaking, current_voice_profile

    if profile is None:
        profile = current_voice_profile

    vp = VOICE_PROFILES.get(profile, VOICE_PROFILES["default"])
    is_speaking = True

    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        # Select voice by index (index 1 = female on most Windows systems)
        v_index = vp["voice_index"] if len(voices) > vp["voice_index"] else 0
        engine.setProperty('voice', voices[v_index].id)
        engine.setProperty('rate', vp["rate"])
        engine.setProperty('volume', vp["volume"])

        print(f"{chatbot_name}: {text}")
        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        tm.sleep(0.4)
        # Flush audio buffer so Usha doesn't hear herself (Set 2 Tuesday)
        with audio_queue.mutex:
            audio_queue.queue.clear()
        is_speaking = False

# ==========================================
# CONTEXT AWARENESS HELPER (Set 2 Wednesday)
# ==========================================
def get_last_bot_topic():
    """Returns the last topic the bot spoke about from history."""
    for role, text in reversed(conversation_history):
        if role == "bot":
            return text
    return ""

def add_to_history(role, text):
    """Thread-safe conversation history management."""
    conversation_history.append((role, text))
    if len(conversation_history) > 20:  # Expanded from 10 for better context (Set 2 Monday)
        conversation_history.pop(0)

# ==========================================
# LOGIC / BRAIN
# Set 4 Monday  — ML intent classification pipeline designed
# Set 4 Tuesday — intents.json dataset created; TF-IDF + Logistic Regression
# Set 4 Wednesday— ml_brain.py module created (modular separation)
# Set 4 Thursday — hybrid ML + rule-based system integrated
# Set 4 Saturday — tested with real voice input; fixes applied
# Set 4 Sunday   — fallback handling for low-confidence ML predictions
# Set 5 Monday   — dataset expanded for better accuracy; model parameters tuned
# Set 5 Tuesday  — confidence threshold enforced inside ml_brain.py
# Set 5 Thursday — full pipeline tested: wake word → STT → ML → memory → TTS
# ==========================================
def get_response(text):
    global user_name, current_voice_profile, memory  # memory must be global to update and save

    text = text.lower().strip()

    # ✅ Always store user input before processing (Set 4 Thursday)
    add_to_history("user", text)

    # 🔥 ML INTENT DETECTION
    # Set 4 Thursday — ML is tried first (hybrid approach)
    # Set 4 Sunday   — try/except ensures fallback to "unknown" on ML failure
    # Set 5 Tuesday  — confidence < 0.6 returns "unknown" inside ml_brain.py
    try:
        intent = predict_intent(text)
    except Exception as e:
        print(f"ML error: {e}")
        intent = "unknown"

    response = ""

    # ==========================================
    # 🔥 ML INTENTS
    # Set 4 Thursday — hybrid: ML runs first, rules are fallback
    # Set 5 Wednesday — name extraction improved with regex (extract_name)
    # If intent is "unknown" (low confidence or error), falls through to rules below
    # ==========================================

    if intent == "set_name":
        name = extract_name(text)  # Set 5 Wednesday — regex extracts full name correctly
        memory["name"] = name
        save_memory(memory)
        response = f"I'll remember that. Your name is {name}"
        add_to_history("bot", response)
        return response

    elif intent == "get_name":
        name = memory.get("name")
        if name:
            response = f"Your name is {name}"
        else:
            response = "I don't know your name yet. Please tell me your name!"
        add_to_history("bot", response)
        return response

    elif intent == "time":
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        response = f"The time is {current_time}"
        add_to_history("bot", response)
        return response

    elif intent == "date":
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        day_of_week = datetime.datetime.now().strftime("%A")
        response = f"Today is {day_of_week}, {current_date}."
        add_to_history("bot", response)
        return response

    elif intent == "joke":
        response = random.choice([
            "Why did the computer get cold? Because it left its Windows open!",
            "Why was the math book sad? Too many problems!",
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fish wearing a bowtie? Sofishticated!",
            "Why did the scarecrow win an award? He was outstanding in his field!"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "wellbeing_ask":
        response = random.choice([
            "I'm doing great, thanks for asking! How about you?",
            "All systems running perfectly! How are you today?",
            "I'm functioning wonderfully! How are things on your end?",
            "I'm fantastic! Hope you're having a wonderful day too.",
            "Doing well and ready to help! How are you?"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "wellbeing_good":
        response = random.choice([
            "That's wonderful to hear! What's making your day good?",
            "Great! I'm glad you're doing well.",
            "Excellent! Keep that positive energy going.",
            "That's fantastic news! Anything exciting happening?",
            "Awesome! Happy to hear that."
        ])
        add_to_history("bot", response)
        return response

    elif intent == "wellbeing_bad":
        last_topic = get_last_bot_topic()
        response = random.choice([
            "I'm sorry to hear that. Would you like to talk about what's bothering you?",
            "Oh no, I'm here for you. What's going on?",
            "That sounds really tough. I'm here to listen if you need to vent.",
            "I'm sorry you're having a hard time. Remember, this too shall pass.",
            "I understand. Sometimes things can be really difficult. Want to share more?"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "encourage":
        response = random.choice([
            "You can do it! Take it one step at a time.",
            "It might be hard right now, but you'll get through it. I believe in you!",
            "Every expert was once a beginner. Don't give up!",
            "You've overcome challenges before and you'll overcome this one too.",
            "Progress, not perfection. You're doing better than you think!"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "greeting":
        name = memory.get("name")
        if name:
            response = random.choice([
                f"Hello {name}! Great to hear from you. How can I help?",
                f"Hi {name}! Lovely to chat with you again.",
                f"Hey {name}! What's on your mind today?"
            ])
        else:
            response = random.choice([
                "Hello there! How can I help you today?",
                "Hi! Lovely to hear from you.",
                "Hey! What brings you here today?"
            ])
        add_to_history("bot", response)
        return response

    elif intent == "farewell":
        name = memory.get("name")
        if name:
            response = random.choice([
                f"Goodbye, {name}! It was great talking with you!",
                f"See you later, {name}! Have a wonderful day!",
                f"Take care, {name}! Come back anytime!"
            ])
        else:
            response = random.choice([
                "Goodbye! It was nice talking to you!",
                "See you later! Have a great day!",
                "Take care! Come back anytime!"
            ])
        add_to_history("bot", response)
        return response

    elif intent == "thanks":
        response = random.choice([
            "You're welcome!", "My pleasure!", "Happy to help!",
            "Anytime!", "Glad I could help!", "No problem at all!"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "apology":
        response = random.choice([
            "No need to apologize! We all make mistakes.",
            "That's perfectly okay!", "No worries at all!",
            "It's completely fine! Don't worry about it."
        ])
        add_to_history("bot", response)
        return response

    elif intent == "compliment_bot":
        response = random.choice([
            "Thank you! That's very kind of you to say.",
            "You're making me blush! Thank you!",
            "That's so sweet of you! I appreciate it.",
            "Thank you! You're pretty awesome yourself!"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "ask_age":
        response = random.choice([
            "Age is just a number! As an AI, I don't have an age in the traditional sense.",
            "I'm ageless! I exist to help and chat whenever you need me.",
            "In AI years, I'm forever young! How about you?"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "ask_creator":
        response = random.choice([
            "I was created by a developer who wanted to make a helpful offline voice assistant!",
            "A programmer built me to be a friendly conversational partner.",
            "I was developed by someone who loves creating helpful AI assistants!"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "ask_name_bot":
        response = random.choice([
            f"My name is {chatbot_name}! Nice to meet you.",
            f"I'm {chatbot_name}, your offline voice assistant!",
            f"You can call me {chatbot_name}! What's your name?"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "ask_capabilities":
        response = (f"I can chat with you, tell jokes, check the time and date, "
                    f"remember your name, change my voice style, and more! "
                    f"Try saying 'tell me a joke' or 'what time is it'.")
        add_to_history("bot", response)
        return response

    elif intent == "help":
        response = random.choice([
            "I'd be happy to help! What do you need assistance with?",
            "I'm here to help! What can I do for you?",
            "Of course! Tell me what you need."
        ])
        add_to_history("bot", response)
        return response

    elif intent == "weather":
        response = random.choice([
            "I don't have real-time weather data, but you can check your local weather app!",
            "Weather updates aren't in my current programming, but I hope it's nice where you are!",
            "For accurate weather, I'd recommend checking a weather service online."
        ])
        add_to_history("bot", response)
        return response

    elif intent == "music":
        response = random.choice([
            "Music is universal! What type of music do you enjoy listening to?",
            "I love the concept of music! It can change moods and bring back memories. What's your favourite genre?",
            "Music is such a powerful form of expression! Who are your favourite artists?"
        ])
        add_to_history("bot", response)
        return response

    elif intent == "voice_faster":
        current_voice_profile = "energetic"
        response = "Sure! Switching to energetic mode. How's this speed for you?"
        add_to_history("bot", response)
        speak(response)
        reset_wake_timer()
        return response

    elif intent == "voice_slower":
        current_voice_profile = "calm"
        response = "Alright, slowing down and staying calm. How's this?"
        add_to_history("bot", response)
        speak(response)
        reset_wake_timer()
        return response

    elif intent == "voice_whisper":
        current_voice_profile = "whisper"
        response = "Okay, switching to a quieter voice for you."
        add_to_history("bot", response)
        speak(response)
        reset_wake_timer()
        return response

    elif intent == "voice_normal":
        current_voice_profile = "default"
        response = "Back to my normal voice! How can I help you?"
        add_to_history("bot", response)
        speak(response)
        reset_wake_timer()
        return response

    # ==========================================
    # RULE-BASED FALLBACK
    # Set 4 Thursday — these rules only run when ML intent is "unknown"
    # or when no ML intent matched above. Ensures reliability.
    # ==========================================

    # ==========================================
    # VOICE PROFILE SWITCHING (Set 2 Sunday / Set 3 Saturday)
    # ==========================================
    if any(phrase in text for phrase in ["speak faster", "talk faster", "be energetic", "sound energetic"]):
        current_voice_profile = "energetic"
        response = "Sure! Switching to energetic mode. How's this speed for you?"

    elif any(phrase in text for phrase in ["speak slower", "talk slower", "calm down", "be calm", "relax your voice"]):
        current_voice_profile = "calm"
        response = "Alright, slowing down and staying calm. How's this?"

    elif any(phrase in text for phrase in ["speak quietly", "whisper", "be quiet", "lower your voice"]):
        current_voice_profile = "whisper"
        response = "Okay, switching to a quieter voice for you."

    elif any(phrase in text for phrase in ["normal voice", "default voice", "reset voice", "speak normally"]):
        current_voice_profile = "default"
        response = "Back to my normal voice! How can I help you?"

    # ==========================================
    # GREETINGS
    # ==========================================
    elif any(word in text for word in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]):
        greetings = [
            "Nice to hear from you! How can I help?",
            "Hello there! What's on your mind?",
            "Hi! Lovely to hear from you.",
            "Hey! What brings you here today?",
            "Greetings! How are you doing?",
            "Hello! Great to talk to you.",
            "Hi there! Hope you're having a good day.",
            "Hey there! What would you like to chat about?",
            "Good day to you! How can I assist?",
            "Hi! Ready for a chat?"
        ]
        response = random.choice(greetings)

    # ==========================================
    # ASKING ABOUT WELL-BEING
    # ==========================================
    elif any(phrase in text for phrase in ["how are you", "how do you do", "how's it going", "how are things", "how have you been"]):
        responses = [
            "I'm doing great, thanks for asking! How about you?",
            "I'm functioning perfectly! How are you feeling today?",
            "All systems operational! How's your day going?",
            "I'm good, just happy to be chatting with you! How about yourself?",
            "I'm excellent! What about you?",
            "Doing well, thank you! How are things on your end?",
            "I'm fantastic! Hope you're having a wonderful day too.",
            "Good and ready to help! How are you?",
            "Doing awesome! How's everything with you?",
            "I'm wonderful! And you?"
        ]
        response = random.choice(responses)

    # ==========================================
    # USER WELL-BEING — POSITIVE
    # ==========================================
    elif any(phrase in text for phrase in ["i'm good", "i am good", "doing well", "i'm fine", "i am fine", "i'm okay", "i am okay"]):
        responses = [
            "That's wonderful to hear!",
            "Great! I'm glad you're doing well.",
            "Excellent! Keep that positive energy going.",
            "That's fantastic news!",
            "Awesome! Happy to hear that.",
            "Wonderful! Hope it stays that way.",
            "Good to know you're feeling good!",
            "That's lovely to hear!",
            "Perfect! What's making your day good?",
            "Nice! Anything exciting happening?"
        ]
        response = random.choice(responses)

    # ==========================================
    # USER WELL-BEING — NEGATIVE (Set 2 Wednesday — context-aware empathy)
    # ==========================================
    elif any(phrase in text for phrase in ["i'm not good", "i am not good", "i'm sad", "i am sad", "feeling down", "not well", "having a bad day"]):
        last_topic = get_last_bot_topic()
        if last_topic:
            responses = [
                "I'm sorry to hear that. Want to talk about what's bothering you?",
                "Oh no, I noticed we were just chatting and things seem harder now. I'm here.",
                "That sounds tough. Would you like to share what's going on?",
                "I'm here for you. Sometimes just talking helps.",
                "Sorry you're having a rough time. What happened?",
            ]
        else:
            responses = [
                "I'm sorry to hear that. Would you like to talk about it?",
                "Oh no, I'm here for you. What's bothering you?",
                "I'm sorry you're feeling that way. Remember, this too shall pass.",
                "That sounds tough. I'm here to listen if you need to vent.",
                "I understand. Sometimes days can be difficult. Want to share what's going on?",
            ]
        response = random.choice(responses)

    # ==========================================
    # INTRODUCTIONS / NAMES
    # ==========================================
    elif any(phrase in text for phrase in ["your name", "who are you", "what are you called"]):
        responses = [
            f"My name is {chatbot_name}! Nice to meet you.",
            f"I'm {chatbot_name}, your voice assistant!",
            f"You can call me {chatbot_name}! What's your name?",
            f"I go by {chatbot_name}. How about you?",
            f"I'm {chatbot_name}! And you are?",
            f"My name is {chatbot_name}. What should I call you?",
            f"I'm {chatbot_name}, pleased to make your acquaintance!",
            f"They call me {chatbot_name}! What's yours?",
            f"{chatbot_name} at your service! And your name is?"
        ]
        response = random.choice(responses)

    elif "my name is" in text or "call me " in text or (
            "i am " in text and not any(p in text for p in ["i am good", "i am fine", "i am okay", "i am not", "i am sad"])):
        if "my name is" in text:
            name = text.split("my name is")[-1].strip()
        elif "call me " in text:
            name = text.split("call me ")[-1].strip()
        elif "i am " in text:
            name = text.split("i am ")[-1].strip()
        else:
            name = "there"

        name = name.split()[0].capitalize() if name.split() else "there"
        user_name = name

        responses = [
            f"Nice to meet you, {name}!",
            f"Hello {name}! That's a lovely name.",
            f"Great to meet you, {name}! How can I help you today?",
            f"Hi {name}! Thanks for introducing yourself.",
            f"Welcome, {name}! I'm {chatbot_name}.",
            f"Pleased to meet you, {name}!",
            f"Hello {name}! What brings you here today?",
            f"Hi there, {name}! Lovely to make your acquaintance.",
            f"Hey {name}! I'll remember that.",
            f"{name}, that's a nice name! How are you today?"
        ]
        response = random.choice(responses)

    # ==========================================
    # THANK YOU
    # ==========================================
    elif any(word in text for word in ["thank", "thanks", "appreciate", "grateful"]):
        responses = [
            "You're welcome!",
            "My pleasure!",
            "Happy to help!",
            "Anytime!",
            "Don't mention it!",
            "You're very welcome!",
            "Glad I could help!",
            "It's my pleasure to assist you!",
            "No problem at all!",
            "You're most welcome!"
        ]
        response = random.choice(responses)

    # ==========================================
    # APOLOGIES
    # ==========================================
    elif any(word in text for word in ["sorry", "apologize", "apology", "my bad"]):
        responses = [
            "No need to apologize! We all make mistakes.",
            "That's perfectly okay!",
            "No worries at all!",
            "It's completely fine!",
            "Don't worry about it!",
            "No problem at all!",
            "All good! No apology needed.",
            "It happens! No need to say sorry.",
            "Water under the bridge!",
            "Absolutely no problem!"
        ]
        response = random.choice(responses)

    # ==========================================
    # TIME AND DATE
    # ==========================================
    elif any(phrase in text for phrase in ["what time", "current time", "time now", "what's the time"]):
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        responses = [
            f"The current time is {current_time}.",
            f"It's {current_time} right now.",
            f"According to my clock, it's {current_time}.",
            f"The time is {current_time}.",
            f"It's currently {current_time}.",
            f"My clock shows {current_time}.",
            f"Right now it's {current_time}.",
            f"Time check: it's {current_time}.",
            f"The time at the moment is {current_time}.",
            f"I see it's {current_time}."
        ]
        response = random.choice(responses)

    elif any(phrase in text for phrase in ["what date", "today's date", "current date", "what day is today"]):
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        day_of_week = datetime.datetime.now().strftime("%A")
        responses = [
            f"Today is {day_of_week}, {current_date}.",
            f"It's {current_date} today.",
            f"Today's date is {current_date}.",
            f"We're on {day_of_week}, {current_date}.",
            f"The date today is {current_date}.",
            f"It's {day_of_week}, {current_date}.",
            f"According to the calendar, it's {current_date}.",
            f"Today is {current_date}.",
            f"The current date is {current_date}.",
            f"It's {day_of_week} today, {current_date}."
        ]
        response = random.choice(responses)

    # ==========================================
    # WEATHER
    # ==========================================
    elif any(word in text for word in ["weather", "temperature", "forecast", "raining", "sunny", "cold", "hot"]):
        responses = [
            "I don't have real-time weather data, but you can check your local weather app!",
            "For accurate weather, I'd recommend checking a weather service.",
            "I wish I could tell you! Unfortunately I don't have that capability yet.",
            "You might want to check a weather website for current conditions.",
            "Weather updates aren't in my programming, but I hope it's nice where you are!",
            "I can't access weather data, but I hope the weather is pleasant for you!",
            "For weather info, you'd need to consult a dedicated weather service.",
            "I'm not connected to weather satellites, but I can chat about other things!",
            "Weather isn't my specialty, but I can help with many other topics!"
        ]
        response = random.choice(responses)

    # ==========================================
    # COMPLIMENTS
    # ==========================================
    elif any(phrase in text for phrase in ["you are nice", "you are helpful", "you are smart", "you are amazing", "you are great", "i like you"]):
        responses = [
            "Thank you! That's very kind of you to say.",
            "You're making me blush! Thank you!",
            "That's so sweet of you! I appreciate it.",
            "Thank you! You're pretty awesome yourself!",
            "You're too kind! I'm just here to help.",
            "Thank you! That means a lot to me.",
            "You're making my day with your kind words!",
            "Thanks! I'm just doing my best to help.",
            "Thank you for the compliment! You're wonderful too.",
            "I appreciate that! It's nice to be appreciated."
        ]
        response = random.choice(responses)

    # ==========================================
    # FAREWELLS
    # ==========================================
    elif any(word in text for word in ["bye", "goodbye", "see you", "farewell", "later", "take care"]):
        if user_name:
            responses = [
                f"Goodbye, {user_name}! It was nice talking to you!",
                f"See you later, {user_name}! Have a great day!",
                f"Take care, {user_name}! Come back anytime!",
                f"Farewell, {user_name}! It was a pleasure chatting!",
                f"Bye, {user_name}! Hope to talk again soon!",
                f"See you soon, {user_name}! Stay safe!",
                f"Goodbye, {user_name}! Thanks for the chat!",
                f"Until next time, {user_name}!",
                f"Take it easy, {user_name}! Bye!",
                f"Catch you later, {user_name}!"
            ]
        else:
            responses = [
                "Goodbye! It was nice talking to you!",
                "See you later! Have a great day!",
                "Take care! Come back anytime!",
                "Farewell! It was a pleasure chatting!",
                "Bye! Hope to talk again soon!",
                "See you soon! Stay safe!",
                "Goodbye! Thanks for the chat!",
                "Until next time!",
                "Take it easy! Bye!",
                "Catch you later!"
            ]
        response = random.choice(responses)

    # ==========================================
    # WHAT CAN YOU DO
    # ==========================================
    elif any(phrase in text for phrase in ["what can you do", "what do you do", "your capabilities", "what are your functions"]):
        responses = [
            "I can chat with you, answer questions, tell jokes, check the time, and keep you company!",
            "I'm here to have conversations, answer your questions, and help pass the time!",
            "I can talk about many things, tell stories, share facts, or just listen if you need someone to talk to!",
            "My main function is conversation! I can discuss topics, share information, or just keep you company.",
            "I'm a voice assistant! I can chat, answer questions, tell jokes, and change my voice style if you ask.",
            "I'm designed for friendly conversation! I can discuss various topics with you.",
            "I'm here to chat! Whether you want to talk about your day, ask questions, or just want company.",
            "I'm a chatbot created for conversation! I can also change how I speak — try asking me to speak faster or slower!"
        ]
        response = random.choice(responses)

    elif any(phrase in text for phrase in ["who made you", "who created you", "who built you", "who developed you"]):
        responses = [
            "I was created by a developer who wanted to make a helpful voice assistant!",
            "A programmer built me to be a friendly conversational partner.",
            "I was developed by someone who loves creating helpful AI assistants!",
            "My creator is a developer who wanted to make conversation more accessible.",
            "I was built by a programmer who enjoys creating useful chatbots!",
            "A developer created me to be a helpful and friendly AI companion.",
            "I was made by someone who believes in the power of friendly conversation!",
            "My creator is a programmer who built me to assist and chat with people.",
        ]
        response = random.choice(responses)

    # ==========================================
    # JOKES
    # ==========================================
    elif any(word in text for word in ["joke", "funny", "make me laugh", "tell me something funny"]):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What do you call a fish wearing a bowtie? Sofishticated!",
            "Why did the bicycle fall over? Because it was two-tired!",
            "What do you call a fake noodle? An impasta!",
            "Why did the math book look so sad? Because it had too many problems!",
            "What's orange and sounds like a parrot? A carrot!",
            "What do you call a sleeping bull? A bulldozer!",
            "Why did the cookie go to the doctor? Because it felt crummy!",
            "Why did the tomato turn red? Because it saw the salad dressing!",
            "What do you get when you cross a snowman and a vampire? Frostbite!"
        ]
        response = random.choice(jokes)

    # ==========================================
    # FOOD AND DRINKS
    # ==========================================
    elif any(word in text for word in ["hungry", "food", "eat", "dinner", "lunch", "breakfast", "meal", "restaurant"]):
        responses = [
            "Food is wonderful! What's your favorite type of cuisine?",
            "I may not eat, but I love hearing about delicious food! What are you craving?",
            "Food brings people together! Do you enjoy cooking or eating out more?",
            "Talking about food makes me wish I could taste! What's the best meal you've ever had?",
            "Food is such an important part of culture! What's your go-to comfort food?",
            "Whether you're cooking or dining out, food is always an adventure!",
            "What are your thoughts on trying exotic foods? Or do you prefer familiar dishes?"
        ]
        response = random.choice(responses)

    elif any(word in text for word in ["coffee", "tea", "drink", "beverage", "juice", "water"]):
        responses = [
            "A good drink can be so refreshing! Do you have a favorite beverage?",
            "I don't drink, but I've heard coffee is many people's morning ritual!",
            "Tea or coffee? That's a classic question! Which do you prefer?",
            "Hydration is important! Water is essential for health.",
            "Some people swear by their morning coffee, others by tea! What's your preference?",
            "Herbal teas can be so soothing! Do you have a favorite kind?",
            "The variety of drinks around the world is amazing! From matcha to espresso to chai..."
        ]
        response = random.choice(responses)

    # ==========================================
    # MUSIC
    # ==========================================
    elif any(word in text for word in ["music", "song", "band", "artist", "concert", "listen to", "playlist"]):
        responses = [
            "Music is universal! What type of music do you enjoy listening to?",
            "I love the concept of music! It can change moods and bring back memories. What's your favorite genre?",
            "Music is such a powerful form of expression! Who are your favorite artists?",
            "Do you play any instruments or just enjoy listening to music?",
            "Music can be so nostalgic! What songs bring back memories for you?",
            "The diversity of music around the world is incredible!",
            "What was the last song you listened to that really moved you?"
        ]
        response = random.choice(responses)

    # ==========================================
    # MOVIES AND TV
    # ==========================================
    elif any(word in text for word in ["movie", "film", "tv show", "netflix", "cinema", "actor", "actress", "watch"]):
        responses = [
            "Movies and TV can be such great entertainment! What's your favorite genre?",
            "What was the last really good movie or show you watched?",
            "Do you enjoy going to the cinema or watching at home more?",
            "Some movies become cultural touchstones! Are there any films you've watched multiple times?",
            "TV shows can create such loyal followings! Have you ever binge-watched a series?",
            "Have you watched anything recently that you'd highly recommend?"
        ]
        response = random.choice(responses)

    # ==========================================
    # BOOKS
    # ==========================================
    elif any(word in text for word in ["book", "read", "novel", "author", "library", "story"]):
        responses = [
            "Books open up entire worlds! What type of books do you enjoy?",
            "Reading is such a wonderful hobby! Do you prefer fiction or non-fiction?",
            "What's the best book you've read recently?",
            "Some books stay with you forever! Are there any that changed your perspective?",
            "Do you have favorite authors or do you like discovering new ones?",
            "E-books, audiobooks, or physical books — do you have a preference?"
        ]
        response = random.choice(responses)

    # ==========================================
    # HOBBIES
    # ==========================================
    elif any(word in text for word in ["hobby", "hobbies", "free time", "weekend", "fun", "enjoy", "passion"]):
        responses = [
            "Hobbies make life so much richer! What do you enjoy doing in your free time?",
            "Having a passion outside work is so important! What are yours?",
            "What's your favorite way to spend a free afternoon?",
            "Do you have any hobbies you've picked up recently?",
            "Some people find their greatest joy in their hobbies! What's yours?",
            "Learning new skills as a hobby can be so rewarding!",
            "What hobby would you like to try if you had more time?"
        ]
        response = random.choice(responses)

    # ==========================================
    # TECHNOLOGY
    # ==========================================
    elif any(word in text for word in ["computer", "phone", "app", "software", "tech", "gadget", "device", "internet"]):
        responses = [
            "Technology moves so fast! What tech are you most interested in?",
            "Do you enjoy learning about new technologies or do you prefer when things stay familiar?",
            "What's your most-used gadget or piece of technology?",
            "Technology can be both helpful and overwhelming! How do you balance it?",
            "What technological advancement has most impacted your life?",
            "Apps and software keep evolving! Do you have favorite apps?",
            "What futuristic technology are you most excited about?"
        ]
        response = random.choice(responses)

    # ==========================================
    # WORK AND STUDY
    # ==========================================
    elif any(word in text for word in ["work", "job", "career", "office", "boss", "colleague", "study", "school", "university", "homework"]):
        responses = [
            "Work and study take up so much of our time! What do you do?",
            "Do you enjoy your work or studies? What do you find most rewarding?",
            "Work-life balance is important! How do you manage it?",
            "Have you faced any interesting challenges at work or school recently?",
            "What skills are most important in your field?",
            "Do you work better alone or as part of a team?",
            "Continuous learning is key! Are you studying anything currently?",
            "Do you have any goals for your career or education?"
        ]
        response = random.choice(responses)

    # ==========================================
    # HEALTH AND WELLNESS
    # ==========================================
    elif any(word in text for word in ["health", "wellness", "fitness", "meditation", "yoga", "sleep", "diet", "mental health"]):
        responses = [
            "Taking care of health is so important! What wellness practices do you follow?",
            "Mental health is just as important as physical health! How do you maintain yours?",
            "Do you have any regular exercise or wellness routines?",
            "Sleep is crucial! How's your sleep schedule?",
            "Nutrition plays a big role in health! Do you pay attention to your diet?",
            "Stress management is key in today's world! What helps you relax?",
            "Have you tried meditation or mindfulness practices?",
            "What's your favorite way to de-stress after a long day?"
        ]
        response = random.choice(responses)

    # ==========================================
    # FAMILY AND RELATIONSHIPS
    # ==========================================
    elif any(word in text for word in ["family", "parent", "sibling", "brother", "sister", "mother", "father", "relationship", "friend", "friendship"]):
        responses = [
            "Family and relationships are so important! Tell me about yours.",
            "Friends can become like family! Do you have close friends?",
            "Family dynamics can be complex but meaningful!",
            "What qualities do you value most in relationships?",
            "Friendships evolve over time! Have you maintained long-term friendships?",
            "Family traditions can be so special! Do you have any?",
            "What's the best relationship advice you've received?",
            "Supportive relationships make such a difference in life!"
        ]
        response = random.choice(responses)

    # ==========================================
    # PETS AND ANIMALS
    # ==========================================
    elif any(word in text for word in ["pet", "dog", "cat", "animal", "puppy", "kitten", "bird", "fish"]):
        responses = [
            "Pets bring so much joy! Do you have any pets?",
            "Animals can be such wonderful companions! Tell me about your pets if you have any.",
            "What's your favorite animal and why?",
            "Do you prefer dogs, cats, or other animals?",
            "Animals teach us about unconditional love!",
            "Pets can be so therapeutic! Do you agree?"
        ]
        response = random.choice(responses)

    # ==========================================
    # CURRENT EVENTS
    # ==========================================
    elif any(word in text for word in ["news", "current events", "headlines", "politics", "world", "society"]):
        responses = [
            "The world is always changing! What current events interest you?",
            "Staying informed is important, but so is managing news consumption!",
            "What topics in the news have caught your attention recently?",
            "Do you follow local, national, or international news more closely?",
            "Balancing being informed with mental wellbeing is important!",
            "What positive news stories have you heard recently?"
        ]
        response = random.choice(responses)

    # ==========================================
    # PHILOSOPHICAL / DEEP
    # ==========================================
    elif any(word in text for word in ["meaning of life", "purpose", "happiness", "success", "dream", "goal", "aspiration"]):
        responses = [
            "These are deep questions! What does happiness mean to you?",
            "Purpose and meaning can be very personal! How do you define success?",
            "Life's big questions don't always have easy answers! What are your thoughts?",
            "Dreams and goals give direction! What are you working toward?",
            "Finding purpose is a journey! What gives your life meaning?",
            "Success can be measured in many ways! How do you measure it?",
            "Goals keep us moving forward! What's your next big goal?"
        ]
        response = random.choice(responses)

    # ==========================================
    # FUN FACTS / TRIVIA
    # ==========================================
    elif any(word in text for word in ["random", "interesting", "fact", "trivia", "learn", "curious"]):
        fun_facts = [
            "Did you know honey never spoils? Archaeologists found honey in Egyptian tombs over 3000 years old — still perfectly good!",
            "Octopuses have three hearts! Two pump blood to the gills, one pumps it to the rest of the body.",
            "A day on Venus is longer than a year on Venus! It takes 243 Earth days to rotate but only 225 to orbit the Sun.",
            "The shortest war in history was between Britain and Zanzibar in 1896 — it lasted just 38 minutes!",
            "Bananas are berries, but strawberries aren't! Botanically, berries have seeds inside.",
            "A group of flamingos is called a flamboyance!",
            "Humans share 50 percent of their DNA with bananas!",
            "The Eiffel Tower can be 15 centimeters taller during summer due to thermal expansion!"
        ]
        response = random.choice(fun_facts)

    # ==========================================
    # HELP REQUESTS
    # ==========================================
    elif any(word in text for word in ["help", "assist", "support", "guide", "advice"]):
        responses = [
            "I'd be happy to help! What do you need assistance with?",
            "I'm here to help! What can I do for you?",
            "How can I assist you today?",
            "I'm ready to help! What do you need?",
            "What kind of help are you looking for?",
            "Happy to provide assistance! Tell me what you need.",
            "I'm here to support you! What's on your mind?"
        ]
        response = random.choice(responses)

    # ==========================================
    # AGREEING
    # ==========================================
    elif any(word in text for word in ["yes", "yeah", "yep", "sure", "absolutely", "definitely", "agree"]):
        responses = [
            "Great! I'm glad we're on the same page.",
            "Absolutely! That makes sense.",
            "I agree! That's a good point.",
            "Yes! I think so too.",
            "Definitely! You're right about that.",
            "Sure thing! I concur.",
            "I'm with you on that!"
        ]
        response = random.choice(responses)

    # ==========================================
    # DISAGREEING
    # ==========================================
    elif any(word in text for word in ["no", "nope", "disagree", "not really", "don't think so"]):
        responses = [
            "That's okay! We can have different perspectives.",
            "I understand your point, even if I see it differently.",
            "No problem! It's good to hear different viewpoints.",
            "That's fair! Everyone has their own opinion.",
            "I respect that! Thanks for sharing your perspective.",
            "No worries! Diversity of thought is valuable."
        ]
        response = random.choice(responses)

    # ==========================================
    # CONFUSED / UNCLEAR
    # ==========================================
    elif any(phrase in text for phrase in ["i don't know", "not sure", "uncertain", "confused", "what do you mean"]):
        responses = [
            "That's okay! Sometimes things aren't clear right away.",
            "No problem! Would you like to explore this topic more?",
            "It's okay to be uncertain! What part is unclear?",
            "Confusion can lead to learning! Would you like to discuss it further?",
            "That's perfectly fine! We can figure it out together."
        ]
        response = random.choice(responses)

    # ==========================================
    # ENCOURAGEMENT
    # ==========================================
    elif any(phrase in text for phrase in ["i can't", "it's hard", "difficult", "struggling", "challenging"]):
        responses = [
            "You can do it! Take it one step at a time.",
            "It might be hard now, but you'll get through it!",
            "Challenges help us grow! I believe in you.",
            "You're stronger than you think! Keep going.",
            "Every expert was once a beginner! Don't give up.",
            "Progress, not perfection! You're doing great.",
            "You've overcome challenges before, and you'll overcome this one too!"
        ]
        response = random.choice(responses)

    # ==========================================
    # PERSONAL ACHIEVEMENTS
    # ==========================================
    elif any(word in text for word in ["achieved", "accomplished", "proud", "success", "milestone", "goal reached"]):
        responses = [
            "That's fantastic! Congratulations on your achievement!",
            "Well done! You should be proud of yourself.",
            "Amazing accomplishment! Your hard work paid off.",
            "Congratulations! That's something to celebrate!",
            "So proud of you! That's a significant milestone.",
            "Excellent work! You deserve to feel accomplished.",
            "Huge congratulations! That's truly impressive."
        ]
        response = random.choice(responses)

    # ==========================================
    # LOVE AND ROMANCE
    # ==========================================
    elif any(word in text for word in ["love", "romance", "dating", "relationship", "partner", "crush"]):
        responses = [
            "Love is a beautiful thing! Tell me more if you'd like.",
            "Relationships can be wonderful and challenging! What's on your mind?",
            "Love comes in many forms! Are you speaking from personal experience?",
            "Romantic relationships can teach us so much about ourselves!",
            "Everyone's love story is unique! What aspect interests you?"
        ]
        response = random.choice(responses)

    # ==========================================
    # AGE QUESTIONS
    # ==========================================
    elif any(phrase in text for phrase in ["how old", "what is your age", "your age"]):
        responses = [
            "Age is just a number! As an AI, I don't have an age in the traditional sense.",
            "I'm ageless! I exist to help and chat whenever you need me.",
            "In AI years, I'm forever young! How about you, if you don't mind me asking?",
            "I don't experience time like humans do! I'm always here when you need me.",
            "My age is irrelevant — what matters is I'm here to help you!"
        ]
        response = random.choice(responses)

    # ==========================================
    # FEELINGS / EMOTIONS (Set 3 Saturday — emotional voice modulation groundwork)
    # ==========================================
    elif any(word in text for word in ["feel", "feeling", "emotion", "emotional", "mood"]):
        feelings = [
            "feel happy", "feel sad", "feel excited", "feel anxious",
            "feel nervous", "feel calm", "feel angry", "feel disappointed",
            "feel grateful", "feel lonely", "feel confident", "feel scared",
            "feel surprised", "feel bored", "feel hopeful", "feel tired"
        ]

        for feeling in feelings:
            if feeling in text:
                emotion = feeling.split()[1]
                responses = [
                    f"It's completely normal to {feeling} sometimes. Want to talk more about it?",
                    f"Thanks for sharing that you {feeling}. How long have you been feeling this way?",
                    f"I hear that you {feeling}. Would you like to explore what's causing this?",
                    f"Feeling {emotion} is part of being human. You're not alone in this.",
                    f"I appreciate you sharing that you {feeling}. How can I support you right now?",
                    f"Everyone experiences feeling {emotion} at times. What usually helps you?",
                    f"Thank you for being open about feeling {emotion}. Would you like to discuss it further?"
                ]
                # Adjust voice tone based on detected emotion (Set 3 Saturday emotional modulation)
                if emotion in ["sad", "lonely", "scared", "disappointed", "anxious"]:
                    speak_profile_override = "calm"
                elif emotion in ["excited", "happy", "hopeful", "confident"]:
                    speak_profile_override = "energetic"
                else:
                    speak_profile_override = current_voice_profile

                response = random.choice(responses)
                add_to_history("bot", response)
                speak(response, profile=speak_profile_override)
                reset_wake_timer()
                return response

    # ==========================================
    # DEFAULT RESPONSES (Set 2 Wednesday — context awareness)
    # ==========================================
    if not response:
        if chatbot_name.lower() in text:
            responses = [
                f"You said my name! What can I do for you?",
                f"Yes, that's me! How can I help?",
                f"You called? I'm listening!",
                f"That's my name! What would you like to talk about?",
                f"I heard my name! What's on your mind?",
                f"I'm here! What would you like to discuss?"
            ]
            response = random.choice(responses)

        elif text.endswith('?') or any(word in text for word in ["what", "why", "how", "when", "where", "who", "which"]):
            responses = [
                "That's an interesting question! What are your thoughts on it?",
                "I'd love to hear your perspective on that first!",
                "That's something worth exploring! What do you think?",
                "Interesting question! I'd be curious to know your opinion.",
                "Good question! What's your take on it?",
                "That's worth discussing! What's your view?"
            ]
            response = random.choice(responses)

        elif any(word in text for word in ["i think", "i believe", "i feel", "in my opinion", "from my perspective"]):
            responses = [
                "Thanks for sharing your perspective! That's interesting.",
                "I appreciate you sharing your thoughts on that!",
                "Thank you for expressing your opinion!",
                "It's great to hear your point of view!",
                "I value hearing your perspective!"
            ]
            response = random.choice(responses)

        else:
            continuations = [
                "Interesting! Tell me more about that.",
                "I see. What happened next?",
                "Go on, I'm listening.",
                "That's fascinating. Please continue.",
                "Really? What else can you tell me?",
                "I understand. What are your thoughts on that?",
                "Interesting point. How does that make you feel?",
                "Tell me more, I'm interested.",
                "I hear you. What else is on your mind?",
                "That's worth discussing further. What's your perspective?",
                "Thanks for sharing that. How did that come about?",
                "I'm following along. What happened then?",
                "That's something to think about. What are your conclusions?",
                "Interesting observation. What led you to that?",
                "I'm curious to know more. Can you elaborate?",
                "That's a good point. How did you arrive at that?",
                "I appreciate you sharing that. What was the context?",
                "That makes sense. What was your reaction?",
                "I see what you mean. What were the circumstances?",
                "Thanks for telling me that. How has that affected you?"
            ]
            response = random.choice(continuations)

    add_to_history("bot", response)
    return response

# ==========================================
# AUDIO PROCESSING
# ==========================================
def audio_callback(indata, frames, time, status):
    """Callback — collects audio only when Usha isn't speaking."""
    if status:
        print(f"Audio status: {status}", end="\r")
    if not is_speaking:
        audio_queue.put(bytes(indata))

def process_audio_queue():
    """
    Main audio processing loop.
    Phase 1: Listen for wake word 'buddy' (passive mode)
    Phase 2: Process commands after wake word (active mode)
    """
    print(f"😴 Passive mode — say '{WAKE_WORD}' to wake {chatbot_name}...")

    while True:
        try:
            audio_data = audio_queue.get(timeout=1.0)

            if recognizer.AcceptWaveform(audio_data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()

                # ---- PASSIVE MODE: ONLY listen for wake word ----
                if not is_awake:
                    if WAKE_WORD in text.lower():
                        activate_wake()
                        speak(f"Yes, I'm here! How can I help you?")
                    continue  # Ignore everything else in passive mode

                # ---- ACTIVE MODE: Process commands ----
                if not passes_confidence_check(text):
                    continue  # Confidence threshold filter (Set 3 Saturday)

                print(f"\nYou: {text}")

                response = get_response(text)
                speak(response)
                reset_wake_timer()  # Reset sleep timer after each interaction

                # Graceful exit on goodbye
                if any(word in text.lower() for word in ["bye", "goodbye", "farewell"]):
                    tm.sleep(1.5)
                    deactivate_wake()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Audio processing error: {e}")
            continue

# ==========================================
# MAIN ENTRY POINT
# ==========================================
def main():
    print(f"\n{'='*55}")
    print(f"  {chatbot_name} — Offline Voice Assistant")
    print(f"{'='*55}")
    print(f"  Wake Word   : '{WAKE_WORD}'")
    print(f"  Sleep Timer : {WAKE_WORD_TIMEOUT} seconds after last interaction")
    print(f"  Voice Modes : default | calm | energetic | whisper")
    print(f"  ML Engine   : TF-IDF + Logistic Regression (ml_brain.py)")
    print(f"  Memory      : memory.json (persistent user data)")
    print(f"  System      : Hybrid ML + Rule-based (Set 4 Thursday)")
    print(f"{'='*55}\n")

    # Start audio processing thread
    processing_thread = threading.Thread(target=process_audio_queue, daemon=True)
    processing_thread.start()

    try:
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=4000,
            dtype="int16",
            channels=1,
            callback=audio_callback
        ) as stream:
            while True:
                tm.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n\n{chatbot_name}: Goodbye! It was nice talking with you!")
    except Exception as e:
        print(f"Stream Error: {e}")
    finally:
        print("\nAssistant stopped.")

if __name__ == "__main__":
    main()