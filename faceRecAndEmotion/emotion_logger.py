"""
Emotion Logger with Telegram Alert System
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
        # TELEGRAM CONFIG (ADD HERE)
        # =========================
        self.TELEGRAM_BOT_TOKEN = "8546984008:AAFzK-kgMe3feW7ob8okmr0Ck0EeZMzblJQ"
        self.TELEGRAM_CHAT_ID = "8706374419"

    # =========================
    # LOG EMOTION (NO CHANGE)
    # =========================
    def log_emotion(self, name, emotion, confidence):
        current_time = datetime.now()

        if name in self.last_log_time:
            diff = (current_time - self.last_log_time[name]).total_seconds()
            if diff < self.log_interval:
                return False

        self.db.log_emotion(name, emotion, confidence)
        self.last_log_time[name] = current_time

        # ALERT CHECK
        self.check_alerts(name, emotion, confidence)

        return True

    # =========================
    # TELEGRAM FUNCTION (ADD THIS)
    # =========================
    def send_telegram(self, message):
        try:
            url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendMessage"

            requests.post(url, data={
                "chat_id": self.TELEGRAM_CHAT_ID,
                "text": message
            })

            print("📨 Telegram sent")

        except Exception as e:
            print("Telegram error:", e)

    # =========================
    # ALERT SYSTEM (MODIFIED)
    # =========================
    def check_alerts(self, name, emotion, confidence):

        if emotion in ["angry", "sad", "fear"] and confidence > 75:

            message = f"""
🚨 Emotion Alert 🚨
Name: {name}
Emotion: {emotion}
Confidence: {confidence}%
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

            print(message)

            # SEND TO TELEGRAM
            self.send_telegram(message)

    # =========================
    # DAILY SUMMARY (NO CHANGE)
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