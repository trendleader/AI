"""
voice_assistant_streamlit.py — Streamlit UI for the Voice Assistant
====================================================================
Root-cause fix for MediaFileStorageError / audio cutoff:
  Streamlit's in-memory media storage GC's audio before the browser fetches it.
  This app injects audio as a base64 data-URI directly into an <audio autoplay>
  HTML element — no Streamlit media file storage involved at all.

TTS: gTTS (Google TTS) — works in any environment (no audio device needed).
STT: browser Web Speech API via JS component, with typed-input fallback.
"""

import os
import base64
import datetime
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit.components.v1 import html as st_html
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("VAStreamlit")

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
HF_API_KEY: str     = os.getenv("HF_API_KEY", "")
LOG_PATH: Path      = Path("conversation_log.json")
MEMORY_WINDOW: int  = 10

HF_GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
HF_EMOTION_MODEL    = "j-hartmann/emotion-english-distilroberta-base"

PERSONALITIES: dict = {
    "ava": {
        "name": "Ava",
        "emoji": "🌸",
        "tts_lang": "en",
        "tts_tld": "com",
        "system_prompt": (
            "You are Ava, a warm, empathetic, and friendly voice assistant. "
            "You speak naturally, like a knowledgeable friend. Acknowledge the "
            "user's emotions when relevant, offer encouragement, and keep answers "
            "concise since your output is spoken aloud. Never use bullet points, "
            "markdown, or special characters."
        ),
        "greeting": "Hello! I'm Ava, your voice assistant. How can I help you today?",
        "color": "#ff6b9d",
    },
    "max": {
        "name": "Max",
        "emoji": "⚡",
        "tts_lang": "en",
        "tts_tld": "co.uk",
        "system_prompt": (
            "You are Max, a professional and highly efficient voice assistant. "
            "Be direct, precise, and results-focused. Keep answers brief and "
            "actionable. Never use bullet points, markdown, or special characters."
        ),
        "greeting": "Max here. Ready to assist. What do you need?",
        "color": "#4a90d9",
    },
    "luna": {
        "name": "Luna",
        "emoji": "🌙",
        "tts_lang": "en",
        "tts_tld": "com.au",
        "system_prompt": (
            "You are Luna, a creative, curious, and imaginative voice assistant. "
            "Approach every question with wonder and offer unexpected perspectives. "
            "Be engaging and fun, but concise. Never use bullet points, markdown, "
            "or special characters."
        ),
        "greeting": "Hey there! Luna here, ready to explore whatever's on your mind!",
        "color": "#9b59b6",
    },
}

EMOTION_RESPONSES: dict = {
    "joy":      "I can hear you're in a great mood! ",
    "sadness":  "I'm sorry to hear you're feeling down. ",
    "anger":    "I understand your frustration. ",
    "fear":     "I hear some concern in what you said. ",
    "surprise": "That does sound surprising! ",
    "disgust":  "I can tell this is bothering you. ",
    "neutral":  "",
}


# ── TTS — gTTS → base64 audio injection (bypasses Streamlit media storage) ───

def _tts_to_base64(text: str, lang: str = "en", tld: str = "com") -> Optional[str]:
    """Convert text to speech and return base64-encoded MP3 bytes."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tts.save(tmp.name)
            with open(tmp.name, "rb") as f:
                audio_bytes = f.read()
        os.unlink(tmp.name)
        return base64.b64encode(audio_bytes).decode()
    except Exception as exc:
        log.error("gTTS error: %s", exc)
        return None


def autoplay_audio(b64: str) -> None:
    """
    Inject audio as a base64 data-URI into an autoplay <audio> element.
    This completely sidesteps Streamlit's media file storage and the
    MediaFileStorageError / audio cutoff bug.
    """
    audio_html = f"""
    <audio autoplay style="display:none">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>
        // Re-trigger playback in case autoplay was blocked
        document.querySelector('audio').play().catch(() => {{}});
    </script>
    """
    st_html(audio_html, height=0)


# ── Emotion detection ─────────────────────────────────────────────────────────

def detect_emotion(text: str, hf_api_key: str) -> str:
    if not hf_api_key:
        return "neutral"
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=hf_api_key)
        result = client.text_classification(text, model=HF_EMOTION_MODEL)
        return result[0].label.lower()
    except Exception as exc:
        log.warning("Emotion detection failed: %s", exc)
        return "neutral"


# ── AI response ───────────────────────────────────────────────────────────────

def get_response_openai(
    query: str, memory: list, system_prompt: str, openai_key: str
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(memory[-MEMORY_WINDOW:])
    messages.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def get_response_huggingface(
    query: str, memory: list, system_prompt: str, hf_key: str
) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=hf_key)
    history = "".join(
        f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}\n"
        for t in memory[-MEMORY_WINDOW:]
    )
    prompt = (
        f"<s>[INST] {system_prompt}\n\n{history}User: {query} [/INST]"
    )
    result = client.text_generation(
        prompt,
        model=HF_GENERATION_MODEL,
        max_new_tokens=300,
        temperature=0.7,
        repetition_penalty=1.1,
    )
    if "[/INST]" in result:
        result = result.split("[/INST]", 1)[-1]
    return result.strip()


def get_ai_response(
    query: str,
    memory: list,
    system_prompt: str,
    backend: str,
    openai_key: str,
    hf_key: str,
    emotion: str = "neutral",
) -> str:
    try:
        if backend == "openai":
            reply = get_response_openai(query, memory, system_prompt, openai_key)
        else:
            reply = get_response_huggingface(query, memory, system_prompt, hf_key)
    except Exception as exc:
        log.error("AI error: %s", exc)
        return "I ran into a problem reaching the AI service. Please try again."
    prefix = EMOTION_RESPONSES.get(emotion, "")
    return f"{prefix}{reply}" if prefix else reply


# ── Persistent log ────────────────────────────────────────────────────────────

def append_to_log(role: str, content: str, personality: str) -> None:
    existing = []
    if LOG_PATH.exists():
        # Try UTF-8 first; fall back to latin-1 for files written by Windows apps
        # (byte 0x92 = Windows-1252 curly apostrophe, invalid in strict UTF-8)
        for enc in ("utf-8", "latin-1"):
            try:
                with LOG_PATH.open("r", encoding=enc) as f:
                    existing = json.load(f)
                break
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError:
                break
    existing.append({
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now().isoformat(),
        "personality": personality,
    })
    with LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ── Web Speech API component (browser microphone) ─────────────────────────────

SPEECH_INPUT_JS = """
<div id="speech-container" style="text-align:center; padding:8px;">
  <button id="mic-btn"
    style="background:#e74c3c;color:white;border:none;border-radius:50%;
           width:56px;height:56px;font-size:24px;cursor:pointer;
           box-shadow:0 2px 6px rgba(0,0,0,.3);">
    🎙️
  </button>
  <p id="status" style="color:#888;font-size:13px;margin-top:6px;">
    Click to speak
  </p>
  <p id="transcript" style="font-weight:bold;font-size:15px;min-height:20px;"></p>
</div>
<script>
(function(){
  const btn = document.getElementById('mic-btn');
  const status = document.getElementById('status');
  const transcript = document.getElementById('transcript');

  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    status.textContent = 'Speech recognition not supported in this browser.';
    btn.disabled = true;
    return;
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const rec = new SpeechRecognition();
  rec.continuous = false;
  rec.interimResults = true;
  rec.lang = 'en-US';

  let listening = false;

  btn.addEventListener('click', () => {
    if (listening) {
      rec.stop();
    } else {
      rec.start();
    }
  });

  rec.onstart = () => {
    listening = true;
    btn.style.background = '#27ae60';
    status.textContent = 'Listening…';
  };

  rec.onresult = (e) => {
    let interim = '', final_t = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) final_t += e.results[i][0].transcript;
      else interim += e.results[i][0].transcript;
    }
    transcript.textContent = final_t || interim;
    if (final_t) {
      // Send result to Streamlit via query param trick
      const url = new URL(window.parent.location.href);
      url.searchParams.set('voice_input', final_t.trim());
      window.parent.history.replaceState({}, '', url.toString());
      // Trigger a small input event so Streamlit picks it up
      window.parent.postMessage({type: 'voice_input', text: final_t.trim()}, '*');
    }
  };

  rec.onend = () => {
    listening = false;
    btn.style.background = '#e74c3c';
    status.textContent = 'Click to speak';
  };

  rec.onerror = (e) => {
    status.textContent = 'Error: ' + e.error;
    listening = false;
    btn.style.background = '#e74c3c';
  };
})();
</script>
"""


# ── Main Streamlit app ────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="wide")

    # ── Session state defaults ─────────────────────────────────────────────
    defaults = {
        "messages": [],
        "memory": [],
        "personality": "ava",
        "backend": "openai",
        "tts_enabled": True,
        "emotion_enabled": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    profile = PERSONALITIES[st.session_state["personality"]]

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        openai_key = st.text_input(
            "OpenAI API Key",
            value=OPENAI_API_KEY,
            type="password",
        )
        hf_key = st.text_input(
            "HuggingFace API Token",
            value=HF_API_KEY,
            type="password",
        )

        st.divider()
        backend = st.radio(
            "AI Backend",
            ["openai", "huggingface"],
            index=0 if st.session_state["backend"] == "openai" else 1,
        )
        st.session_state["backend"] = backend

        st.divider()
        st.subheader("🎭 Personality")
        for key, p in PERSONALITIES.items():
            if st.button(f"{p['emoji']} {p['name']}", use_container_width=True):
                if st.session_state["personality"] != key:
                    st.session_state["personality"] = key
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": p["greeting"],
                        "emotion": "neutral",
                    })
                    if st.session_state["tts_enabled"]:
                        b64 = _tts_to_base64(p["greeting"], p["tts_lang"], p["tts_tld"])
                        if b64:
                            st.session_state["pending_audio"] = b64
                    st.rerun()

        st.divider()
        st.session_state["tts_enabled"] = st.toggle("🔊 Text-to-Speech", value=st.session_state["tts_enabled"])
        st.session_state["emotion_enabled"] = st.toggle("💡 Emotion Detection", value=st.session_state["emotion_enabled"])

        if st.button("🧹 Clear conversation"):
            st.session_state["messages"] = []
            st.session_state["memory"] = []
            st.rerun()

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        f"<h1 style='color:{profile['color']}'>"
        f"{profile['emoji']} {profile['name']} — Voice Assistant"
        f"</h1>",
        unsafe_allow_html=True,
    )
    st.caption(f"Backend: **{st.session_state['backend'].upper()}** | "
               f"TTS: **{'on' if st.session_state['tts_enabled'] else 'off'}** | "
               f"Emotion: **{'on' if st.session_state['emotion_enabled'] else 'off'}**")

    # ── Play any pending audio (from personality switch etc.) ──────────────
    if "pending_audio" in st.session_state:
        autoplay_audio(st.session_state.pop("pending_audio"))

    # ── Greeting on first load ─────────────────────────────────────────────
    if not st.session_state["messages"]:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": profile["greeting"],
            "emotion": "neutral",
        })

    # ── Render chat history ────────────────────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("emotion") and msg["emotion"] not in ("neutral", ""):
                st.caption(f"Emotion detected: {msg['emotion']}")

    # ── Input: typed chat + mic section ───────────────────────────────────
    col_chat, col_mic = st.columns([3, 1])

    with col_chat:
        typed = st.chat_input("Type your message…")

    with col_mic:
        with st.expander("🎙️ Voice Input", expanded=False):
            st_html(SPEECH_INPUT_JS, height=130)
            voice_typed = st.text_input(
                "Transcript (edit if needed):",
                key="voice_transcript",
                placeholder="Spoken text appears here…",
            )
            send_voice = st.button("Send voice input", use_container_width=True)

    # Determine which input to act on
    user_query: Optional[str] = None
    if typed:
        user_query = typed.strip()
    elif send_voice and voice_typed.strip():
        user_query = voice_typed.strip()

    if user_query:
        api_key = openai_key if st.session_state["backend"] == "openai" else hf_key
        if not api_key:
            st.error("Please enter your API key in the sidebar.")
            st.stop()

        # Show user message
        st.session_state["messages"].append({"role": "user", "content": user_query, "emotion": ""})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Detect emotion
        emotion = "neutral"
        if st.session_state["emotion_enabled"] and hf_key:
            with st.spinner("Detecting emotion…"):
                emotion = detect_emotion(user_query, hf_key)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner(f"{profile['name']} is thinking…"):
                reply = get_ai_response(
                    query=user_query,
                    memory=st.session_state["memory"],
                    system_prompt=profile["system_prompt"],
                    backend=st.session_state["backend"],
                    openai_key=openai_key,
                    hf_key=hf_key,
                    emotion=emotion,
                )
            st.markdown(reply)
            if emotion != "neutral":
                st.caption(f"Emotion detected: {emotion}")

        # TTS — inject as base64 data-URI (no Streamlit media storage)
        if st.session_state["tts_enabled"]:
            with st.spinner("Generating speech…"):
                b64 = _tts_to_base64(reply, profile["tts_lang"], profile["tts_tld"])
            if b64:
                autoplay_audio(b64)
                # Also show a persistent audio player so the user can replay
                audio_bytes = base64.b64decode(b64)
                st.audio(audio_bytes, format="audio/mp3")

        # Update state
        st.session_state["messages"].append({"role": "assistant", "content": reply, "emotion": emotion})
        st.session_state["memory"].append({"role": "user", "content": user_query})
        st.session_state["memory"].append({"role": "assistant", "content": reply})
        if len(st.session_state["memory"]) > MEMORY_WINDOW * 2:
            st.session_state["memory"] = st.session_state["memory"][-MEMORY_WINDOW * 2:]

        append_to_log("user", user_query, st.session_state["personality"])
        append_to_log("assistant", reply, st.session_state["personality"])


if __name__ == "__main__":
    main()
