"""
Emotion Logger + Telegram Command System + Graph Report
Optimized version with non-blocking threads to prevent main video loop lag
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force headless rendering to prevent main-thread GUI popups
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import time
import threading  # Added for asynchronous network updates

from faceRecAndEmotion.database_manager import DatabaseManager
from faceRecAndEmotion.config import EMOTION_LOG_INTERVAL



class EmotionLogger:
    def __init__(self):
        self.db = DatabaseManager()
        self.log_interval = EMOTION_LOG_INTERVAL
        self.last_log_time = {}

        # TELEGRAM CONFIG
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

        print("✅ Emotion Logger initialized with async threading capabilities")

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

        # Check alerts asynchronously to keep the main video feed smooth
        threading.Thread(target=self.check_alerts, args=(name, emotion, confidence), daemon=True).start()
        return True

    # =========================
    # TELEGRAM TEXT (Non-blocking worker)
    # =========================
    def send_text(self, chat_id, message, run_async=True):
        def _send():
            try:
                url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendMessage"
                requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
            except Exception as e:
                print(f"⚠️ Telegram text alert failed: {e}")

        if run_async:
            threading.Thread(target=_send, daemon=True).start()
        else:
            _send()

    # =========================
    # TELEGRAM PHOTO (Non-blocking worker)
    # =========================
    def send_photo(self, chat_id, image_path, run_async=True):
        def _send():
            try:
                url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendPhoto"
                with open(image_path, "rb") as photo:
                    requests.post(url, data={"chat_id": chat_id}, files={"photo": photo}, timeout=10)
            except Exception as e:
                print(f"⚠️ Telegram photo transfer failed: {e}")

        if run_async:
            threading.Thread(target=_send, daemon=True).start()
        else:
            _send()

    # =========================
    # ALERT SYSTEM
    # =========================
    def check_alerts(self, name, emotion, confidence):
        alert_emotions = ["angry", "sad", "fear", "unknown"]

        if emotion in alert_emotions and confidence > 75:
            message = f"""🚨 EMOTION ALERT 🚨

👤 Name: {name}
🙂 Emotion: {emotion.upper()}
📊 Confidence: {confidence:.1f}%
🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

            # Send via internal async routine
            self.send_text(self.TELEGRAM_CHAT_ID, message, run_async=False)

    # =========================
    # GRAPH GENERATOR + SEND
    # =========================
    def generate_report(self, name):
        try:
            logs = self.db.emotion_logs.get("logs", [])
            df = pd.DataFrame(logs)

            if df.empty:
                print("NO LOGS AVAILABLE")
                return None, None

            # Standardize string entries to prevent mismatched strings
            df["name"] = df["name"].astype(str).str.strip().str.lower()
            name_clean = name.strip().lower()

            person_df = df[df["name"] == name_clean]
            print("FOUND ROWS:", len(person_df))

            if person_df.empty:
                return None, None

            emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral", "unknown"]
            counts = person_df["emotion"].value_counts().reindex(emotions, fill_value=0)

            # Generate plot cleanly using a locked thread structure
            fig, ax = plt.subplots(figsize=(7, 4.5))
            counts.plot(kind="bar", color=['#4CAF50', '#2196F3', '#F44336', '#9C27B0', '#00BCD4', '#9E9E9E', '#FF5722'], ax=ax)

            ax.set_title(f"Emotion Report - {name.capitalize()}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Emotion Category")
            ax.set_ylabel("Occurrence Count")
            plt.tight_layout()

            file_path = f"{name_clean}_report.png"
            fig.savefig(file_path, dpi=100)
            plt.close(fig)

            print("GRAPH SAVED:", file_path)
            return file_path, counts
        except Exception as e:
            print(f"⚠️ Error generating analytics report: {e}")
            return None, None

    # =========================
    # COMMAND LISTENER (Background Loop)
    # =========================
    def listen_telegram(self):
        """
        Polls for inbound user commands.
        Call this using a separate background thread inside your core execution script:
        Example: threading.Thread(target=logger.listen_telegram, daemon=True).start()
        """
        print("🚀 Telegram Command Listener is active in background thread...")
        offset = None

        while True:
            try:
                url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/getUpdates?timeout=5"
                if offset:
                    url += f"&offset={offset}"

                res = requests.get(url, timeout=10).json()

                for update in res.get("result", []):
                    offset = update["update_id"] + 1

                    if "message" not in update:
                        continue

                    chat_id = update["message"]["chat"]["id"]
                    text = update["message"].get("text", "").strip()

                    # ======================
                    # /report Command handling
                    # ======================
                    if text.startswith("/report"):
                        parts = text.split()

                        if len(parts) < 2:
                            self.send_text(chat_id, "Usage: /report Name", run_async=False)
                            continue

                        target_name = parts[1]
                        file_path, counts = self.generate_report(target_name)

                        if file_path is None:
                            self.send_text(chat_id, f"No data logs found for target profile: '{target_name}'", run_async=False)
                            continue

                        # Send file resources synchronously inside the background thread context
                        self.send_photo(chat_id, file_path, run_async=False)

                        summary = f"""📊 Emotion Report - {target_name.capitalize()}

🙂 Happy: {counts['happy']}
😢 Sad: {counts['sad']}
😡 Angry: {counts['angry']}
😨 Fear: {counts['fear']}
😮 Surprise: {counts['surprise']}
😐 Neutral: {counts['neutral']}
❓ Unknown: {counts['unknown']}"""

                        self.send_text(chat_id, summary, run_async=False)

            except Exception as e:
                print(f"⚠️ Listener Connection Warning: {e}")

            time.sleep(1.5)
