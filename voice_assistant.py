"""
voice_assistant.py — Enhanced Voice Assistant (Ava)
=====================================================
Fixes vs original:
  • hf_client always initialized when HF_API_KEY is set (was only init'd for HF backend,
    causing AttributeError in detect_emotion when using OpenAI backend)
  • TTS engine auto-reinitializes after a crash instead of silently failing
  • speak() saves audio to a temp file then plays it — prevents pyttsx3 cutoffs on long text
  • _handle_command() return value properly used (was partially duplicated in run())
  • detect_emotion guarded by hasattr check — safe even if hf_client not available
"""

import os
import sys
import json
import logging
import datetime
import tempfile
from pathlib import Path
from typing import Optional

import speech_recognition as sr
import pyttsx3
from langdetect import detect, DetectorFactory
from openai import OpenAI
from huggingface_hub import InferenceClient

DetectorFactory.seed = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("VoiceAssistant")

# ── Config (env vars or defaults) ─────────────────────────────────────────────
OPENAI_API_KEY: str     = os.getenv("OPENAI_API_KEY", "")
HF_API_KEY: str         = os.getenv("HF_API_KEY", "")
AI_BACKEND: str         = os.getenv("AI_BACKEND", "openai")
HF_GENERATION_MODEL     = "mistralai/Mistral-7B-Instruct-v0.1"
HF_EMOTION_MODEL        = "j-hartmann/emotion-english-distilroberta-base"
MEMORY_WINDOW: int      = 10
LOG_PATH: Path          = Path("conversation_log.json")

# ── Personality profiles ───────────────────────────────────────────────────────
PERSONALITIES: dict = {
    "ava": {
        "name": "Ava",
        "description": "Warm, empathetic, and conversational",
        "system_prompt": (
            "You are Ava, a warm, empathetic, and friendly voice assistant. "
            "You speak naturally, like a knowledgeable friend. You acknowledge "
            "the user's emotions when relevant, offer encouragement, and keep "
            "answers concise since your output is spoken aloud. Never use "
            "bullet points, markdown, or special characters in your responses."
        ),
        "voice_rate": 150,
        "voice_volume": 1.0,
        "greeting": "Hello! I'm Ava, your voice assistant. How can I help you today?",
    },
    "max": {
        "name": "Max",
        "description": "Professional, direct, and efficient",
        "system_prompt": (
            "You are Max, a professional and highly efficient voice assistant. "
            "You are direct, precise, and results-focused. Keep answers brief "
            "and actionable. Avoid filler phrases. Never use bullet points, "
            "markdown, or special characters since your output is spoken aloud."
        ),
        "voice_rate": 170,
        "voice_volume": 0.9,
        "greeting": "Max here. Ready to assist. What do you need?",
    },
    "luna": {
        "name": "Luna",
        "description": "Creative, curious, and imaginative",
        "system_prompt": (
            "You are Luna, a creative, curious, and imaginative voice assistant. "
            "You approach every question with wonder and often offer unexpected "
            "perspectives or creative ideas. Keep answers engaging and fun, but "
            "concise since your output is spoken aloud. Never use bullet points, "
            "markdown, or special characters in your responses."
        ),
        "voice_rate": 140,
        "voice_volume": 1.0,
        "greeting": "Hey there! Luna here, ready to explore whatever's on your mind!",
    },
}

EMOTION_RESPONSES: dict = {
    "joy":      "I can hear that you're in a great mood! ",
    "sadness":  "I'm sorry to hear you're feeling down. ",
    "anger":    "I understand your frustration. ",
    "fear":     "I hear some concern in what you said. ",
    "surprise": "That does sound surprising! ",
    "disgust":  "I can tell this is bothering you. ",
    "neutral":  "",
}


class VoiceAssistant:
    def __init__(self, personality: str = "ava", backend: str = AI_BACKEND) -> None:
        # ── Personality ───────────────────────────────────────────────────
        if personality not in PERSONALITIES:
            log.warning("Unknown personality '%s'. Falling back to 'ava'.", personality)
            personality = "ava"
        self.profile = PERSONALITIES[personality]
        self.personality_key = personality

        # ── Backend ───────────────────────────────────────────────────────
        self.backend = backend.lower()
        if self.backend == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for the OpenAI backend.")
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            log.info("Backend: OpenAI (gpt-4o-mini)")
        elif self.backend == "huggingface":
            if not HF_API_KEY:
                raise ValueError("HF_API_KEY is required for the HuggingFace backend.")
            self.hf_client = InferenceClient(token=HF_API_KEY)
            log.info("Backend: HuggingFace (%s)", HF_GENERATION_MODEL)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'openai' or 'huggingface'.")

        # Always init HF client for emotion detection (regardless of text-gen backend)
        if not hasattr(self, "hf_client") and HF_API_KEY:
            self.hf_client = InferenceClient(token=HF_API_KEY)

        # ── TTS ───────────────────────────────────────────────────────────
        self._init_tts_engine()

        # ── Memory ────────────────────────────────────────────────────────
        self.memory: list = []
        self.log_path = LOG_PATH
        self._load_persistent_log()

        # ── Speech Recognition ────────────────────────────────────────────
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.8
        self.recognizer.energy_threshold = 300

        log.info("VoiceAssistant ready. Personality=%s Backend=%s",
                 self.profile["name"], self.backend)

    def _init_tts_engine(self) -> None:
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", self.profile["voice_rate"])
        self.engine.setProperty("volume", self.profile["voice_volume"])

    # ── Persistent memory ─────────────────────────────────────────────────────

    def _load_persistent_log(self) -> None:
        if self.log_path.exists():
            try:
                with self.log_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memory = data[-MEMORY_WINDOW:]
                log.info("Loaded %d prior turns.", len(self.memory))
            except (json.JSONDecodeError, KeyError):
                log.warning("Could not parse log. Starting fresh.")
                self.memory = []

    def _save_persistent_log(self, role: str, content: str) -> None:
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
            "personality": self.personality_key,
        }
        existing: list = []
        if self.log_path.exists():
            try:
                with self.log_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        existing.append(entry)
        with self.log_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def _update_memory(self, role: str, content: str) -> None:
        self.memory.append({"role": role, "content": content})
        if len(self.memory) > MEMORY_WINDOW:
            self.memory = self.memory[-MEMORY_WINDOW:]
        self._save_persistent_log(role, content)

    # ── Speech Recognition ────────────────────────────────────────────────────

    def listen(self) -> Optional[str]:
        with sr.Microphone() as source:
            print("\n[Listening…]")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)
            except sr.WaitTimeoutError:
                log.warning("Mic timeout — no speech detected.")
                return None
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            log.warning("Speech not understood.")
            return None
        except sr.RequestError as exc:
            log.error("Google Speech API error: %s", exc)
            return None

    # ── Language Detection ────────────────────────────────────────────────────

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            lang = detect(text)
            log.info("Language: %s", lang)
            return lang
        except Exception:
            return "en"

    def _set_tts_language(self, lang_code: str) -> None:
        voices = self.engine.getProperty("voices")
        for voice in voices:
            if lang_code.lower() in voice.id.lower():
                self.engine.setProperty("voice", voice.id)
                log.info("TTS voice → %s", voice.name)
                return
        log.info("No TTS voice for '%s'. Keeping current.", lang_code)

    # ── Emotion Recognition ───────────────────────────────────────────────────

    def detect_emotion(self, text: str) -> str:
        if not hasattr(self, "hf_client"):
            return "neutral"
        try:
            result = self.hf_client.text_classification(text, model=HF_EMOTION_MODEL)
            top = result[0].label.lower()
            log.info("Emotion: %s (%.2f)", top, result[0].score)
            return top
        except Exception as exc:
            log.warning("Emotion detection failed: %s", exc)
            return "neutral"

    # ── AI Response Generation ────────────────────────────────────────────────

    def _get_response_openai(self, query: str) -> str:
        messages = [{"role": "system", "content": self.profile["system_prompt"]}]
        messages.extend(self.memory)
        messages.append({"role": "user", "content": query})
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def _get_response_huggingface(self, query: str) -> str:
        history_text = "".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}\n"
            for t in self.memory
        )
        prompt = (
            f"<s>[INST] {self.profile['system_prompt']}\n\n"
            f"{history_text}User: {query} [/INST]"
        )
        result = self.hf_client.text_generation(
            prompt,
            model=HF_GENERATION_MODEL,
            max_new_tokens=300,
            temperature=0.7,
            repetition_penalty=1.1,
        )
        if "[/INST]" in result:
            result = result.split("[/INST]", 1)[-1]
        return result.strip()

    def get_ai_response(self, query: str, emotion: str = "neutral") -> str:
        if not query:
            return "I didn't catch that. Could you say that again?"
        try:
            if self.backend == "openai":
                reply = self._get_response_openai(query)
            else:
                reply = self._get_response_huggingface(query)
        except Exception as exc:
            log.error("AI response error: %s", exc)
            return "I ran into a problem reaching the AI service. Please try again."
        prefix = EMOTION_RESPONSES.get(emotion, "")
        return f"{prefix}{reply}" if prefix else reply

    # ── Text-to-Speech ────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Save speech to a temp WAV file then play it.
        This prevents pyttsx3 from cutting off long responses because
        runAndWait() on some backends returns before audio finishes streaming.
        """
        print(f"\n{self.profile['name']}: {text}")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()
            # Play the saved file so the full audio renders before we continue
            _play_wav(tmp_path)
        except Exception as exc:
            log.error("TTS error: %s — reinitializing engine.", exc)
            self._init_tts_engine()
            # Fallback: direct say() as last resort
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                log.error("TTS fallback also failed.")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Utility ───────────────────────────────────────────────────────────────

    def switch_personality(self, name: str) -> None:
        key = name.strip().lower()
        if key not in PERSONALITIES:
            self.speak(
                f"I don't know a personality called {name}. "
                f"Available options are: {', '.join(PERSONALITIES)}."
            )
            return
        self.profile = PERSONALITIES[key]
        self.personality_key = key
        self.engine.setProperty("rate", self.profile["voice_rate"])
        self.engine.setProperty("volume", self.profile["voice_volume"])
        log.info("Personality → %s", self.profile["name"])
        self.speak(f"Switching to {self.profile['name']} mode. {self.profile['greeting']}")

    def _handle_command(self, query: str) -> Optional[str]:
        """
        Returns 'exit' to quit, 'handled' if a command ran, None to pass to AI.
        """
        q = query.lower().strip()
        if any(word in q for word in ("exit", "quit", "goodbye", "bye", "stop")):
            self.speak("Goodbye! Have a wonderful day!")
            return "exit"
        for key in PERSONALITIES:
            if key in q and any(t in q for t in ("switch", "change", "become", "use")):
                self.switch_personality(key)
                return "handled"
        if "clear memory" in q or "forget everything" in q:
            self.memory = []
            self.speak("Memory cleared. Starting fresh!")
            return "handled"
        return None

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.speak(self.profile["greeting"])
        while True:
            user_input = self.listen()
            if user_input is None:
                self.speak("I didn't catch that. Please try again.")
                continue

            cmd = self._handle_command(user_input)
            if cmd == "exit":
                break
            if cmd == "handled":
                continue

            lang = self.detect_language(user_input)
            self._set_tts_language(lang)

            emotion = "neutral"
            if hasattr(self, "hf_client"):
                emotion = self.detect_emotion(user_input)

            response = self.get_ai_response(user_input, emotion)
            self.speak(response)
            self._update_memory("user", user_input)
            self._update_memory("assistant", response)


# ── Platform-specific WAV playback ────────────────────────────────────────────
def _play_wav(path: str) -> None:
    """Play a WAV file using the platform's native audio player."""
    import platform
    system = platform.system()
    try:
        if system == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        elif system == "Darwin":
            os.system(f"afplay '{path}'")
        else:
            # Linux: try aplay, then paplay
            if os.system(f"aplay -q '{path}' 2>/dev/null") != 0:
                os.system(f"paplay '{path}' 2>/dev/null")
    except Exception as exc:
        log.warning("WAV playback failed: %s", exc)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Voice Assistant")
    parser.add_argument("--personality", choices=list(PERSONALITIES), default="ava")
    parser.add_argument("--backend", choices=["openai", "huggingface"], default=AI_BACKEND)
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("   VOICE ASSISTANT — Enhanced Edition")
    print(f"   Personality : {args.personality.upper()}")
    print(f"   Backend     : {args.backend.upper()}")
    print(f"   Memory      : rolling {MEMORY_WINDOW}-turn window + {LOG_PATH}")
    print("═" * 60)
    print("  Commands:")
    print("    'exit' / 'quit'           → shut down")
    print("    'switch to max/luna/ava'  → change personality")
    print("    'clear memory'            → wipe conversation history")
    print("═" * 60 + "\n")

    assistant = VoiceAssistant(personality=args.personality, backend=args.backend)
    assistant.run()


if __name__ == "__main__":
    main()
