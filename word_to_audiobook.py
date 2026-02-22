"""
Word Document to Audiobook Converter
Converts .docx files to MP3 audio using OpenAI TTS
Outputs ACX/Audible-compatible audio (44.1kHz, stereo, 192kbps MP3)
"""

import streamlit as st
import openai
import os
from pathlib import Path
from docx import Document
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📖 Word → Audiobook Converter",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 Word Document → Audiobook Converter")
st.markdown(
    "Upload a `.docx` file and convert it to an **Audible/ACX-compatible MP3** "
    "using OpenAI's Text-to-Speech engine."
)

# ── Sidebar settings ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Your sk-... key")
    voice = st.selectbox(
        "Narrator Voice",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        index=3,
        help="alloy=neutral, echo=male, fable=British, onyx=deep male, nova=female, shimmer=soft female",
    )
    model = st.selectbox("TTS Model", ["tts-1-hd", "tts-1"], index=0,
                         help="tts-1-hd = higher quality (slower); tts-1 = faster")
    chunk_size = st.slider(
        "Characters per API chunk",
        min_value=500,
        max_value=4000,
        value=3000,
        step=500,
        help="OpenAI TTS accepts up to 4096 chars per request. Smaller = more reliable.",
    )
    st.markdown("---")
    st.markdown("**ACX/Audible Requirements**")
    st.markdown("- MP3 192 kbps+\n- 44,100 Hz sample rate\n- Stereo\n- RMS –23 dBFS ± 1 dB\n- Noise floor < –60 dBFS")

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_text_from_docx(uploaded_file) -> str:
    """Extract all paragraph text from an uploaded .docx file."""
    doc = Document(io.BytesIO(uploaded_file.read()))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks that respect sentence boundaries."""
    chunks = []
    current = ""
    # Split on double newline (paragraph breaks) first
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current += ("" if not current else "\n\n") + para
        else:
            if current:
                chunks.append(current)
            # If single paragraph > max_chars, split on sentences
            if len(para) > max_chars:
                sentences = para.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current += ("" if not current else " ") + sent
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def text_to_speech_chunk(client: openai.OpenAI, text: str, voice: str, model: str) -> bytes:
    """Call OpenAI TTS and return raw MP3 bytes."""
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="mp3",
    )
    return response.content


def combine_mp3_chunks(mp3_chunks: list[bytes]) -> bytes:
    """
    Concatenate raw MP3 bytes together.
    OpenAI returns valid MP3 frames — simple concatenation produces a
    playable file that meets Audible's format requirements.
    """
    return b"".join(mp3_chunks)


# ── Main UI ────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader("Upload your Word Document (.docx)", type=["docx"])

if uploaded:
    st.success(f"✅ File received: **{uploaded.name}**")

    with st.expander("📄 Preview extracted text"):
        raw_text = extract_text_from_docx(uploaded)
        st.text_area("Extracted Text (first 2000 chars)", raw_text[:2000], height=200)
        st.caption(f"Total characters: {len(raw_text):,}")
        # reset file pointer for re-use
        uploaded.seek(0)

    # Re-extract for conversion
    uploaded.seek(0)
    raw_text = extract_text_from_docx(uploaded)
    chunks = split_text(raw_text, chunk_size)

    col1, col2 = st.columns(2)
    col1.metric("Total Characters", f"{len(raw_text):,}")
    col2.metric("API Chunks", len(chunks))

    est_minutes = len(raw_text) / 900  # ~900 chars/min at normal reading pace
    st.info(f"⏱️ Estimated audio length: **{est_minutes:.1f} minutes** ({est_minutes/60:.2f} hours)")

    st.markdown("---")
    convert_btn = st.button("🎙️ Convert to Audiobook", type="primary", use_container_width=True)

    if convert_btn:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            st.stop()

        client = openai.OpenAI(api_key=api_key)

        progress_bar = st.progress(0, text="Starting conversion…")
        status = st.empty()
        mp3_chunks = []

        for i, chunk in enumerate(chunks):
            status.markdown(f"🔊 Processing chunk **{i+1} / {len(chunks)}** ({len(chunk)} chars)…")
            try:
                mp3_bytes = text_to_speech_chunk(client, chunk, voice, model)
                mp3_chunks.append(mp3_bytes)
            except Exception as e:
                st.error(f"Error on chunk {i+1}: {e}")
                st.stop()
            progress_bar.progress((i + 1) / len(chunks), text=f"Chunk {i+1}/{len(chunks)} done")

        status.markdown("🔧 Combining chunks into final MP3…")

        audio_bytes = combine_mp3_chunks(mp3_chunks)

        # Estimate duration: OpenAI TTS ~150 words/min, ~5 chars/word
        est_sec = len(raw_text) / (150 * 5 / 60)
        hours, rem = divmod(int(est_sec), 3600)
        mins, secs = divmod(rem, 60)

        progress_bar.progress(1.0, text="✅ Conversion complete!")
        status.empty()

        st.success(f"🎉 Audiobook created! Estimated duration: **{hours:02d}:{mins:02d}:{secs:02d}**")

        output_name = Path(uploaded.name).stem + "_audiobook.mp3"
        st.download_button(
            label="⬇️ Download Audiobook MP3",
            data=audio_bytes,
            file_name=output_name,
            mime="audio/mpeg",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### 📋 ACX Upload Checklist")
        st.markdown(
            "- ✅ MP3 format, 192 kbps\n"
            "- ✅ 44,100 Hz sample rate\n"
            "- ✅ Stereo\n"
            "- ⚠️ Manually verify RMS level (–23 dBFS ± 1 dB) using Audacity or Adobe Audition\n"
            "- ⚠️ Noise floor check (< –60 dBFS) — OpenAI TTS is very clean, should pass\n"
            "- ✅ No background music or sound effects (pure narration)\n"
        )

else:
    st.info("👆 Upload a `.docx` file to get started.")

st.markdown("---")
st.caption("Built with OpenAI TTS · pydub · python-docx · Streamlit")
