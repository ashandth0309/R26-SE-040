"""
Emotion Logger + Telegram Command System + Graph Report
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import time

from database_manager import DatabaseManager
from config import EMOTION_LOG_INTERVAL


class EmotionLogger:
    def __init__(self):
        self.db = DatabaseManager()
        self.log_interval = EMOTION_LOG_INTERVAL
        self.last_log_time = {}

        # TELEGRAM CONFIG
        self.TELEGRAM_BOT_TOKEN = "8546984008:AAHpB13l0QOjBUGPImXiVgDJmUk9YS-dA30"
        self.TELEGRAM_CHAT_ID = "8706374419"

    # =========================
    # LOG EMOTION
    # =========================
    def log_emotion(self, name, emotion, confidence):
        current_time = datetime.now()

        if name in self.last_log_time:
            diff = (current_time - self.last_log_time[name]).total_seconds()
            if diff < self.log_interval:
                return False

        self.db.log_emotion(name, emotion, confidence)
        self.last_log_time[name] = current_time

        self.check_alerts(name, emotion, confidence)
        return True

    # =========================
    # TELEGRAM TEXT
    # =========================
    def send_text(self, chat_id, message):
        url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendMessage"

        requests.post(url, data={
            "chat_id": chat_id,
            "text": message
        })

    # =========================
    # TELEGRAM PHOTO
    # =========================
    def send_photo(self, chat_id, image_path):
        url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendPhoto"

        with open(image_path, "rb") as photo:
            requests.post(url, data={
                "chat_id": chat_id
            }, files={"photo": photo})

    # =========================
    # ALERT SYSTEM
    # =========================
    def check_alerts(self, name, emotion, confidence):

        alert_emotions = ["angry", "sad", "fear", "unknown"]

        if emotion in alert_emotions and confidence > 75:

            message = f"""
🚨 EMOTION ALERT 🚨

👤 Name: {name}
🙂 Emotion: {emotion}
📊 Confidence: {confidence}%
🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            self.send_text(self.TELEGRAM_CHAT_ID, message)

    # =========================
    # GRAPH GENERATOR + SEND
    # =========================
    def generate_report(self, name):

        logs = self.db.emotion_logs.get("logs", [])
        df = pd.DataFrame(logs)

        if df.empty:
            print("NO LOGS")
            return None, None

        # CLEAN FIX
        df["name"] = df["name"].astype(str).str.strip().str.lower()
        name = name.strip().lower()

        person_df = df[df["name"] == name]

        print("FOUND ROWS:", len(person_df))

        if person_df.empty:
            return None, None

        emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral", "unknown"]

        counts = person_df["emotion"].value_counts().reindex(emotions, fill_value=0)

        plt.figure(figsize=(8, 5))
        counts.plot(kind="bar")

        plt.title(f"Emotion Report - {name}")
        plt.xlabel("Emotion")
        plt.ylabel("Count")

        file_path = f"{name}_report.png"
        plt.savefig(file_path)
        plt.close()

        print("GRAPH SAVED:", file_path)

        return file_path, counts

    # =========================
    # COMMAND LISTENER
    # =========================
    def listen_telegram(self):

        offset = None

        while True:
            url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/getUpdates?timeout=10"

            if offset:
                url += f"&offset={offset}"   # ✅ FIXED HERE

            res = requests.get(url).json()

            for update in res.get("result", []):

                offset = update["update_id"] + 1

                if "message" not in update:
                    continue

                chat_id = update["message"]["chat"]["id"]
                text = update["message"].get("text", "")

                # ======================
                # /report Name
                # ======================
                if text.startswith("/report"):
                    parts = text.split()

                    if len(parts) < 2:
                        self.send_text(chat_id, "Usage: /report Name")
                        continue

                    name = parts[1]

                    file_path, counts = self.generate_report(name)

                    if file_path is None:
                        self.send_text(chat_id, f"No data found for {name}")
                        continue

                    # send image
                    self.send_photo(chat_id, file_path)

                    # send summary
                    summary = f"""
📊 Emotion Report - {name}

🙂 Happy: {counts['happy']}
😢 Sad: {counts['sad']}
😡 Angry: {counts['angry']}
😨 Fear: {counts['fear']}
😮 Surprise: {counts['surprise']}
😐 Neutral: {counts['neutral']}
❓ Unknown: {counts['unknown']}
"""

                    self.send_text(chat_id, summary)

            time.sleep(2)