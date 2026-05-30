import speech_recognition as sr
from openai import OpenAI
from gtts import gTTS
import pygame
import time
import os

# 1. Setup Local AI and Audio Player
# We use the openai library, but point it to your local Ollama server!
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama' # This is required by the library, but Ollama ignores it
)

pygame.mixer.init()

# 2. The Memory Array
conversation_history = [
    {"role": "system", "content": "You are a friendly, conversational AI. Keep your answers brief, natural, and conversational. Do not use lists or markdown, just talk like a human."}
]

def listen():
    """Listens to the microphone and converts speech to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Listening... (Speak now)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            print("Processing speech...")
            
            # Using Google's free speech recognition
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

def get_brain_response(user_text):
    """Sends the whole conversation to your local Llama 3 model."""
    conversation_history.append({"role": "user", "content": user_text})
    
    try:
        # Ask LOCAL Llama 3 for the response
        response = client.chat.completions.create(
            model="llama3", # Switched from gpt-3.5-turbo to llama3
            messages=conversation_history,
            max_tokens=100 
        )
        ai_text = response.choices[0].message.content
        
        conversation_history.append({"role": "assistant", "content": ai_text})
        
        print(f"🤖 AI: {ai_text}")
        return ai_text
        
    except Exception as e:
        print(f"⚠️ Error thinking: {e}")
        print("Did you remember to keep the Ollama app running in the background?")
        return "I'm having trouble connecting to my brain."

def speak(text):
    """Converts text into a voice using free Google TTS."""
    try:
        # Generate the audio file using free gTTS
        tts = gTTS(text=text, lang='en')
        filename = "temp_response.mp3"
        tts.save(filename)
        
        # Play the audio
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        # Clean up
        pygame.mixer.music.unload()
        os.remove(filename)
        
    except Exception as e:
        print(f"⚠️ Error speaking: {e}")

# --- THE MAIN LOOP ---
def start_conversation():
    print("========================================")
    print("🦙 Llama 3 Voice Bot Started! (100% FREE)")
    print("Press Ctrl+C in the terminal to stop.")
    print("========================================")
    
    greeting = "Hello! I'm your local Llama 3 assistant. What's on your mind?"
    print(f"🤖 AI: {greeting}")
    speak(greeting)
    conversation_history.append({"role": "assistant", "content": greeting})

    while True:
        try:
            user_input = listen()
            
            if user_input:
                ai_reply = get_brain_response(user_input)
                speak(ai_reply)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    start_conversation()