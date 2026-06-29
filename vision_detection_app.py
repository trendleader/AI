import base64
import io
import json
import os
import re

import anthropic
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Detection AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.detection-box {
    border: 2px solid #4CAF50;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    background: rgba(76,175,80,0.08);
}
.face-box {
    border: 2px solid #2196F3;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    background: rgba(33,150,243,0.08);
}
.object-tag {
    display: inline-block;
    background: #4CAF50;
    color: white;
    border-radius: 12px;
    padding: 2px 10px;
    margin: 3px;
    font-size: 0.85rem;
    font-weight: 600;
}
.face-tag {
    display: inline-block;
    background: #2196F3;
    color: white;
    border-radius: 12px;
    padding: 2px 10px;
    margin: 3px;
    font-size: 0.85rem;
    font-weight: 600;
}
.confidence-high { color: #4CAF50; font-weight: 700; }
.confidence-med  { color: #FF9800; font-weight: 700; }
.confidence-low  { color: #F44336; font-weight: 700; }
.stat-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin: 6px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("current_image", None),
    ("detection_results", None),
    ("history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ────────────────────────────────────────────────────────────────────
def image_to_base64(img_bytes: bytes) -> str:
    return base64.standard_b64encode(img_bytes).decode("utf-8")


def to_jpeg_bytes(uploaded) -> bytes:
    img = Image.open(uploaded)
    if img.mode not in ("RGB",):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def api_key_ok() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def confidence_class(conf: float) -> str:
    if conf >= 0.8:
        return "confidence-high"
    if conf >= 0.5:
        return "confidence-med"
    return "confidence-low"


def run_object_detection(img_bytes: bytes, model: str, detail_level: str) -> dict:
    """Ask Claude to detect and list all objects in the image."""
    client = get_client()
    b64 = image_to_base64(img_bytes)

    detail_instruction = {
        "Basic": "List the main objects you can see.",
        "Standard": "List all visible objects with their approximate location (top/center/bottom, left/center/right) and confidence.",
        "Detailed": "List every object with: name, count, location (normalized 0-1 coordinates if possible), confidence, color/size/state attributes.",
    }.get(detail_level, "List all visible objects.")

    prompt = f"""Analyze this image for object detection.

{detail_instruction}

Respond ONLY with valid JSON in this exact format:
{{
  "objects": [
    {{
      "name": "person",
      "count": 2,
      "confidence": 0.95,
      "location": "center-left",
      "attributes": ["standing", "wearing blue shirt"]
    }}
  ],
  "scene_description": "Brief one-sentence scene summary.",
  "total_objects": 5,
  "dominant_objects": ["person", "car"]
}}

Be precise and return only the JSON object, no markdown."""

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def run_facial_analysis(img_bytes: bytes, model: str, privacy_mode: bool) -> dict:
    """Ask Claude to detect faces and analyze facial attributes."""
    client = get_client()
    b64 = image_to_base64(img_bytes)

    if privacy_mode:
        attrs = "only count faces and describe general demographics (approximate age range, expression)"
    else:
        attrs = "describe expression, approximate age range, gender presentation, head pose, eye contact, any visible accessories (glasses, hat, etc.), and emotion"

    prompt = f"""Analyze this image for facial detection and recognition.

For each face detected, {attrs}.

Respond ONLY with valid JSON in this exact format:
{{
  "face_count": 2,
  "faces": [
    {{
      "id": 1,
      "location": "left side of image",
      "confidence": 0.92,
      "expression": "smiling",
      "approximate_age_range": "25-35",
      "emotion": "happy",
      "attributes": ["glasses", "looking at camera"]
    }}
  ],
  "group_summary": "Brief description of the people/group in this image.",
  "faces_detected": true
}}

If no faces are present, set face_count to 0, faces to [], and faces_detected to false.
Return only the JSON object, no markdown."""

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def run_combined_analysis(img_bytes: bytes, model: str, custom_query: str = "") -> dict:
    """Run both object detection and facial analysis together."""
    client = get_client()
    b64 = image_to_base64(img_bytes)

    extra = f"\nAdditional user query: {custom_query}" if custom_query else ""

    prompt = f"""Perform comprehensive image analysis including object detection and facial recognition.{extra}

Respond ONLY with valid JSON:
{{
  "objects": [
    {{"name": "...", "count": 1, "confidence": 0.9, "location": "...", "attributes": []}}
  ],
  "faces": [
    {{"id": 1, "location": "...", "confidence": 0.9, "expression": "...", "approximate_age_range": "...", "emotion": "...", "attributes": []}}
  ],
  "face_count": 0,
  "total_objects": 0,
  "scene_description": "...",
  "scene_type": "indoor/outdoor/...",
  "lighting": "bright/dim/natural/artificial",
  "image_quality": "high/medium/low",
  "notable_insights": ["insight 1", "insight 2"],
  "safety_concerns": []
}}

Return only the JSON object."""

    response = client.messages.create(
        model=model,
        max_tokens=3000,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    for block in response.content:
        if block.type == "text":
            raw = block.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)

    raise ValueError("No text block in response")


def run_custom_query(img_bytes: bytes, model: str, query: str) -> str:
    """Run an arbitrary natural-language query on the image."""
    client = get_client()
    b64 = image_to_base64(img_bytes)

    for chunk in client.messages.stream(
        model=model,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": query},
                ],
            }
        ],
    ) as stream:
        return stream.get_final_text()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("👁️ Vision Detection AI")
    st.caption("Object, Image & Facial Detection powered by Claude")
    st.divider()

    st.subheader("API Configuration")
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        placeholder="sk-ant-...",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    model = st.selectbox(
        "Claude Model",
        ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        index=0,
    )

    st.divider()
    st.subheader("Detection Options")
    detail_level = st.selectbox("Object Detection Detail", ["Basic", "Standard", "Detailed"], index=1)
    privacy_mode = st.toggle("Privacy Mode (faces)", value=False, help="Limits facial attribute analysis to general demographics only")

    st.divider()
    if st.session_state.history:
        st.subheader(f"History ({len(st.session_state.history)})")
        for h in reversed(st.session_state.history[-5:]):
            st.caption(f"📷 {h['name']} — {h['objects']} obj / {h['faces']} faces")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.caption("Powered by Claude Opus 4.8 Vision")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("👁️ Vision Detection AI")
st.caption("Upload any image for AI-powered object detection, facial recognition, and scene analysis")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Quick Scan", "📦 Object Detection", "👤 Facial Analysis", "💬 Custom Query"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUICK SCAN (combined)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Quick Scan — Full Analysis")
    st.caption("Upload an image to instantly detect objects and faces in one pass.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="qs_upload",
        )
        if uploaded:
            img_bytes = to_jpeg_bytes(uploaded)
            st.session_state.current_image = img_bytes
            img_pil = Image.open(io.BytesIO(img_bytes))
            st.image(img_pil, caption=uploaded.name, use_container_width=True)
            w, h = img_pil.size
            st.caption(f"Resolution: {w}×{h} px | Size: {len(img_bytes)//1024} KB")

        custom_q = st.text_input(
            "Optional: additional question",
            placeholder="e.g. Are there any safety hazards?",
        )

        scan_btn = st.button(
            "🔍 Run Full Analysis",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.current_image,
        )

    with col_right:
        if scan_btn:
            if not api_key_ok():
                st.error("Enter your Anthropic API key in the sidebar first.")
            else:
                with st.spinner("Analyzing image with Claude…"):
                    try:
                        results = run_combined_analysis(
                            st.session_state.current_image, model, custom_q
                        )
                        st.session_state.detection_results = results

                        # Save to history
                        st.session_state.history.append(
                            {
                                "name": uploaded.name if uploaded else "image",
                                "objects": results.get("total_objects", len(results.get("objects", []))),
                                "faces": results.get("face_count", 0),
                            }
                        )
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        results = st.session_state.detection_results
        if results:
            # Scene overview
            st.subheader("Scene Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Objects", results.get("total_objects", len(results.get("objects", []))))
            c2.metric("Faces", results.get("face_count", 0))
            c3.metric("Scene", results.get("scene_type", "—"))
            c4.metric("Quality", results.get("image_quality", "—").title())

            st.markdown(f"**Scene:** {results.get('scene_description', '')}")
            st.markdown(f"**Lighting:** {results.get('lighting', '—').title()}")

            # Objects
            objs = results.get("objects", [])
            if objs:
                st.subheader("Detected Objects")
                tags_html = "".join(
                    f'<span class="object-tag">{o["name"]} ({o.get("count",1)})</span>'
                    for o in objs
                )
                st.markdown(tags_html, unsafe_allow_html=True)

                for obj in objs:
                    conf = obj.get("confidence", 0)
                    cls = confidence_class(conf)
                    attrs = ", ".join(obj.get("attributes", []))
                    st.markdown(
                        f'<div class="detection-box">'
                        f'<b>{obj["name"]}</b> ×{obj.get("count",1)} '
                        f'@ {obj.get("location","—")} '
                        f'— <span class="{cls}">{int(conf*100)}% conf</span>'
                        + (f"<br><small>{attrs}</small>" if attrs else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

            # Faces
            faces = results.get("faces", [])
            if faces:
                st.subheader("Detected Faces")
                for face in faces:
                    conf = face.get("confidence", 0)
                    cls = confidence_class(conf)
                    attrs = ", ".join(face.get("attributes", []))
                    st.markdown(
                        f'<div class="face-box">'
                        f'<b>Face #{face["id"]}</b> @ {face.get("location","—")} '
                        f'— <span class="{cls}">{int(conf*100)}% conf</span><br>'
                        f'Expression: {face.get("expression","—")} | '
                        f'Emotion: {face.get("emotion","—")} | '
                        f'Age: {face.get("approximate_age_range","—")}'
                        + (f"<br><small>{attrs}</small>" if attrs else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                if results.get("group_summary") or results.get("faces", []):
                    gs = results.get("group_summary", "")
                    if gs:
                        st.caption(f"Group summary: {gs}")

            # Insights
            insights = results.get("notable_insights", [])
            if insights:
                st.subheader("Notable Insights")
                for ins in insights:
                    st.info(ins)

            concerns = results.get("safety_concerns", [])
            if concerns:
                st.subheader("Safety Concerns")
                for c in concerns:
                    st.warning(c)
        elif not scan_btn:
            st.info("Upload an image and click **Run Full Analysis** to begin.")
            st.markdown(
                """
**What this detects:**
- 📦 Objects — name, count, location, confidence
- 👤 Faces — expression, emotion, age range, attributes
- 🏞️ Scene type, lighting, and image quality
- 💡 Notable insights and safety concerns
"""
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OBJECT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Object Detection")
    st.caption("Dedicated object detection with configurable detail levels.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        obj_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="obj_upload",
        )
        obj_bytes = None
        if obj_file:
            obj_bytes = to_jpeg_bytes(obj_file)
            st.image(Image.open(io.BytesIO(obj_bytes)), caption=obj_file.name, use_container_width=True)
        elif st.session_state.current_image:
            obj_bytes = st.session_state.current_image
            st.image(Image.open(io.BytesIO(obj_bytes)), caption="From Quick Scan", use_container_width=True)
        else:
            st.info("Upload an image or use one from Quick Scan tab.")

        detect_btn = st.button(
            "📦 Detect Objects",
            type="primary",
            use_container_width=True,
            disabled=obj_bytes is None,
        )

    with col_right:
        if detect_btn and obj_bytes:
            if not api_key_ok():
                st.error("Enter your Anthropic API key in the sidebar first.")
            else:
                with st.spinner(f"Running {detail_level} object detection…"):
                    try:
                        res = run_object_detection(obj_bytes, model, detail_level)

                        st.subheader("Detection Results")
                        objs = res.get("objects", [])

                        c1, c2 = st.columns(2)
                        c1.metric("Total Objects", res.get("total_objects", len(objs)))
                        c2.metric("Unique Types", len(set(o["name"] for o in objs)))

                        st.markdown(f"**Scene:** {res.get('scene_description','')}")

                        dom = res.get("dominant_objects", [])
                        if dom:
                            tags = "".join(f'<span class="object-tag">{d}</span>' for d in dom)
                            st.markdown(f"**Dominant:** {tags}", unsafe_allow_html=True)

                        st.divider()
                        for obj in objs:
                            conf = obj.get("confidence", 0)
                            cls = confidence_class(conf)
                            attrs = ", ".join(obj.get("attributes", []))
                            st.markdown(
                                f'<div class="detection-box">'
                                f'<b>{obj["name"]}</b> ×{obj.get("count",1)} '
                                f'@ <em>{obj.get("location","—")}</em> '
                                f'— <span class="{cls}">{int(conf*100)}%</span>'
                                + (f"<br><small>Attributes: {attrs}</small>" if attrs else "")
                                + "</div>",
                                unsafe_allow_html=True,
                            )

                        if not objs:
                            st.info("No objects detected in this image.")

                    except Exception as e:
                        st.error(f"Detection failed: {e}")
        else:
            st.info("Upload an image and click **Detect Objects**.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FACIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Facial Detection & Analysis")
    st.caption("Detect faces and analyze expressions, emotions, and attributes.")

    if privacy_mode:
        st.info("Privacy Mode is ON — facial analysis limited to general demographics.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        face_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="face_upload",
        )
        face_bytes = None
        if face_file:
            face_bytes = to_jpeg_bytes(face_file)
            st.image(Image.open(io.BytesIO(face_bytes)), caption=face_file.name, use_container_width=True)
        elif st.session_state.current_image:
            face_bytes = st.session_state.current_image
            st.image(Image.open(io.BytesIO(face_bytes)), caption="From Quick Scan", use_container_width=True)
        else:
            st.info("Upload an image or use one from Quick Scan tab.")

        face_btn = st.button(
            "👤 Analyze Faces",
            type="primary",
            use_container_width=True,
            disabled=face_bytes is None,
        )

    with col_right:
        if face_btn and face_bytes:
            if not api_key_ok():
                st.error("Enter your Anthropic API key in the sidebar first.")
            else:
                with st.spinner("Analyzing faces…"):
                    try:
                        res = run_facial_analysis(face_bytes, model, privacy_mode)

                        face_count = res.get("face_count", 0)
                        st.metric("Faces Detected", face_count)

                        if not res.get("faces_detected", False) or face_count == 0:
                            st.info("No faces detected in this image.")
                        else:
                            gs = res.get("group_summary", "")
                            if gs:
                                st.markdown(f"**Summary:** {gs}")

                            st.divider()
                            for face in res.get("faces", []):
                                conf = face.get("confidence", 0)
                                cls = confidence_class(conf)
                                attrs = ", ".join(face.get("attributes", []))
                                st.markdown(
                                    f'<div class="face-box">'
                                    f'<b>Face #{face["id"]}</b> — '
                                    f'<span class="{cls}">{int(conf*100)}% confidence</span><br>'
                                    f'📍 Location: {face.get("location","—")}<br>'
                                    f'😊 Expression: {face.get("expression","—")}<br>'
                                    f'💭 Emotion: {face.get("emotion","—")}<br>'
                                    f'👤 Age range: {face.get("approximate_age_range","—")}'
                                    + (f"<br>🏷️ {attrs}" if attrs else "")
                                    + "</div>",
                                    unsafe_allow_html=True,
                                )

                            # Emotion distribution chart
                            emotions = [
                                f.get("emotion", "unknown")
                                for f in res.get("faces", [])
                                if f.get("emotion")
                            ]
                            if len(emotions) > 1:
                                from collections import Counter
                                import plotly.graph_objects as go

                                counts = Counter(emotions)
                                fig = go.Figure(
                                    go.Bar(
                                        x=list(counts.keys()),
                                        y=list(counts.values()),
                                        marker_color="#2196F3",
                                    )
                                )
                                fig.update_layout(
                                    title="Emotion Distribution",
                                    xaxis_title="Emotion",
                                    yaxis_title="Count",
                                    margin=dict(l=0, r=0, t=40, b=0),
                                    height=280,
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Facial analysis failed: {e}")
        else:
            st.info("Upload an image and click **Analyze Faces**.")
            st.markdown(
                """
**Facial analysis includes:**
- Face count and location
- Expression and emotion
- Approximate age range
- Accessories (glasses, hat, etc.)
- Group summary

Enable **Privacy Mode** in the sidebar to limit analysis.
"""
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOM QUERY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Custom Query")
    st.caption("Ask anything about your image — streamed in real time.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        cq_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="cq_upload",
        )
        cq_bytes = None
        if cq_file:
            cq_bytes = to_jpeg_bytes(cq_file)
            st.image(Image.open(io.BytesIO(cq_bytes)), caption=cq_file.name, use_container_width=True)
        elif st.session_state.current_image:
            cq_bytes = st.session_state.current_image
            st.image(Image.open(io.BytesIO(cq_bytes)), caption="From Quick Scan", use_container_width=True)
        else:
            st.info("Upload an image or use one from the Quick Scan tab.")

        presets = [
            "How many people are in this image?",
            "What objects pose a safety risk?",
            "Describe the emotions of people in this image.",
            "What brand logos or text can you read?",
            "What time of day does this appear to be?",
            "Describe this image in detail for someone who can't see it.",
            "What is the approximate age and gender of each person?",
            "Are there any animals in this image? Describe them.",
            "Custom…",
        ]
        preset_q = st.selectbox("Quick questions", presets)
        if preset_q == "Custom…":
            query = st.text_area("Your question", placeholder="Ask anything about this image…", height=100)
        else:
            query = preset_q
            st.text_area("Your question", value=query, height=60, disabled=True)

        ask_btn = st.button(
            "💬 Ask Claude",
            type="primary",
            use_container_width=True,
            disabled=not (cq_bytes and query),
        )

    with col_right:
        st.subheader("Claude's Answer")
        if ask_btn:
            if not api_key_ok():
                st.error("Enter your Anthropic API key in the sidebar first.")
            elif not cq_bytes:
                st.warning("Upload an image first.")
            elif not query.strip():
                st.warning("Enter a question first.")
            else:
                box = st.empty()
                full_text = ""
                with st.spinner("Thinking…"):
                    try:
                        client = get_client()
                        b64 = image_to_base64(cq_bytes)
                        with client.messages.stream(
                            model=model,
                            max_tokens=2000,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/jpeg",
                                                "data": b64,
                                            },
                                        },
                                        {"type": "text", "text": query},
                                    ],
                                }
                            ],
                        ) as stream:
                            for text_chunk in stream.text_stream:
                                full_text += text_chunk
                                box.markdown(full_text + "▌")
                        box.markdown(full_text)
                    except Exception as e:
                        st.error(f"Query failed: {e}")
        else:
            st.info("Upload an image, pick or type a question, then click **Ask Claude**.")
            st.markdown(
                """
**Example questions:**
- "How many people are in this image?"
- "What safety hazards are visible?"
- "What brand logos can you see?"
- "Describe the emotions of each person."
- "What objects are on the table?"
- "Is there text visible? Read it."
"""
            )
