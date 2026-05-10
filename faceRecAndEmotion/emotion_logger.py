"""
Emotion Logger with Telegram Alert System (Upgraded)
- Added UNKNOWN alert support
- Cleaner alert logic
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests

from database_manager import DatabaseManager
from config import EMOTION_LOG_INTERVAL


class EmotionLogger:
    def __init__(self):
        self.db = DatabaseManager()
        self.log_interval = EMOTION_LOG_INTERVAL
        self.last_log_time = {}

        # =========================
        # TELEGRAM CONFIG
        # =========================
        self.TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
        self.TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

    # =========================
    # LOG EMOTION
    # =========================
    def log_emotion(self, name, emotion, confidence):
        current_time = datetime.now()

        # anti spam per user
        if name in self.last_log_time:
            diff = (current_time - self.last_log_time[name]).total_seconds()
            if diff < self.log_interval:
                return False

        self.db.log_emotion(name, emotion, confidence)
        self.last_log_time[name] = current_time

        # check alerts
        self.check_alerts(name, emotion, confidence)

        return True

    # =========================
    # TELEGRAM SENDER
    # =========================
    def send_telegram(self, message):
        try:
            url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendMessage"

            response = requests.post(url, data={
                "chat_id": self.TELEGRAM_CHAT_ID,
                "text": message
            })

            print("Telegram Response:", response.text)

        except Exception as e:
            print("Telegram error:", e)

    # =========================
    # ALERT SYSTEM (UPGRADED)
    # =========================
    def check_alerts(self, name, emotion, confidence):

        alert_emotions = ["angry", "sad", "fear", "unknown"]

        if emotion in alert_emotions and confidence > 75:

            # special unknown handling
            if emotion == "unknown":
                message = (
                    f"⚠️ UNKNOWN EMOTION ALERT ⚠️\n\n"
                    f"👤 Name: {name}\n"
                    f"❓ Emotion: UNKNOWN (not clearly detected)\n"
                    f"📊 Confidence: {confidence}%\n"
                    f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                message = (
                    f"🚨 EMOTION ALERT 🚨\n\n"
                    f"👤 Name: {name}\n"
                    f"🙂 Emotion: {emotion}\n"
                    f"📊 Confidence: {confidence}%\n"
                    f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

            print(message)
            self.send_telegram(message)

    # =========================
    # DAILY SUMMARY
    # =========================
    def get_daily_summary(self):
        logs = self.db.emotion_logs.get("logs", [])

        if not logs:
            return None

        df = pd.DataFrame(logs)

        return {
            "total_detections": len(logs),
            "emotion_counts": df["emotion"].value_counts().to_dict(),
            "average_confidence": float(df["confidence"].mean()),
            "most_common_emotion": df["emotion"].mode()[0]
        }