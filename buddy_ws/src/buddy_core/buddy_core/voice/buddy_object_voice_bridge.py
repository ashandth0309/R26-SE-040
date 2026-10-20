"""Non-motion voice dialogue adapter for BUDDY's existing Flask vision API."""
import json
import os
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from buddy_core.voice.dialogue_manager import DialogueReply

class ObjectConversation:
    def __init__(self, dialogue, url=None):
        self.dialogue = dialogue
        self.url = (url or os.getenv("BUDDY_OBJECT_API_URL", "http://127.0.0.1:5000")).rstrip("/")
        self.teaching_offered = False

    def api(self, method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.url + path, data=data, method=method,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=4) as response:
                return json.load(response)
        except (HTTPError, URLError, OSError, TimeoutError, ValueError):
            return None

    @staticmethod
    def reply(text, context):
        context.add_user("[object request]")
        context.add_buddy(text, "object_learning")
        return DialogueReply(text, "object_vision", "object_learning")

    def respond(self, text, context):
        phrase = re.sub(r"[^a-z0-9' ]", " ", text.lower())
        phrase = " ".join(phrase.split())
        phrase = re.sub(r"^(?:hey |hello )?buddy ", "", phrase)
        state = self.api("GET", "/api/objects/teach/status")
        phase = state.get("phase", "idle") if isinstance(state, dict) else "idle"
        active = bool(state and state.get("active"))

        def teach(action, name=""):
            data = self.api("POST", "/api/objects/teach",
                            {"action": action, "name": name})
            message = ("Object learning is unavailable right now." if data is None
                       else str(data.get("reply") or data.get("error")
                                or "That action could not be completed."))
            return self.reply(message, context)

        if active:
            if phrase in ("cancel", "cancel teaching", "cancel learning", "never mind"):
                return teach("cancel")
            if phase in ("confirm_save", "confirm_duplicate"):
                if phrase in ("yes", "yes please", "yes remember it",
                              "remember it", "save it", "yes save it"):
                    return teach("confirm")
                if phrase in ("no", "no thanks", "don't save it", "different"):
                    return teach("reject")
                return self.reply("Please say yes or no.", context)
            return self.reply("I'm collecting six views. Turn the object slowly.", context)

        named = re.fullmatch(r"(?:this is|it is|it's|that is|that's) (?:my |a |an |the )?(.+)", phrase)
        if named and 1 <= len(named.group(1)) <= 60:
            self.teaching_offered = False
            return teach("start", named.group(1))

        if self.teaching_offered and phrase in ("yes", "yes please"):
            self.teaching_offered = False
            return self.reply("What should I call it? Say, this is my pen.", context)

        if self.teaching_offered and phrase in ("no", "no thanks"):
            self.teaching_offered = False
            return self.reply("Okay, I won't learn that object.", context)

        if phrase in ("what is this", "what's this", "what is that",
                      "identify this", "identify this object",
                      "can you identify this"):
            data = self.api("POST", "/api/objects/identify", {})
            if data is None:
                return self.reply("The camera recognition service is unavailable.", context)
            self.teaching_offered = bool(data.get("offer_teach"))
            return self.reply(str(data.get("text") or "I couldn't identify that."), context)

        return self.dialogue.respond(text, context)
