"""
avatar_component.py
Generates the self-contained HTML string for the animated interviewer panel.
Import this and call render_avatar_panel(persona, message, speaking) to get
the HTML to inject via st.components.v1.html().
"""

def get_avatar_html(persona: str, message: str, is_speaking: bool = False) -> str:
    """
    persona  : "behavioral" | "technical" | "hiring"
    message  : The text the interviewer just said (will be spoken aloud)
    is_speaking: controls the speaking animation state on load
    """

    # ── Persona configs ────────────────────────────────────────────────────────
    personas = {
        "behavioral": {
            "name":    "Sarah Chen",
            "title":   "HR & Behavioral Coach",
            "skin":    "#F4C2A1",
            "hair":    "#2C1810",
            "shirt":   "#7C3AED",
            "accent":  "#A78BFA",
            "bg_from": "#1e1a2e",
            "bg_to":   "#2d1f3d",
            "badge":   "#7C3AED",
            "voice_pitch": "1.1",
            "voice_rate":  "0.92",
        },
        "technical": {
            "name":    "Marcus Rivera",
            "title":   "Senior Technical Interviewer",
            "skin":    "#C8956C",
            "hair":    "#1A1A1A",
            "shirt":   "#0D6EFD",
            "accent":  "#60A5FA",
            "bg_from": "#0f1e2e",
            "bg_to":   "#1a2d40",
            "badge":   "#0D6EFD",
            "voice_pitch": "0.85",
            "voice_rate":  "0.88",
        },
        "hiring": {
            "name":    "Diana Wells",
            "title":   "Hiring Manager",
            "skin":    "#E8B89A",
            "hair":    "#5C3D1E",
            "shirt":   "#0F766E",
            "accent":  "#2DD4BF",
            "bg_from": "#0f2020",
            "bg_to":   "#1a3030",
            "badge":   "#0F766E",
            "voice_pitch": "1.05",
            "voice_rate":  "0.90",
        },
    }

    p = personas.get(persona, personas["behavioral"])
    # Escape message for JS string
    safe_msg = message.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: linear-gradient(135deg, {p['bg_from']} 0%, {p['bg_to']} 100%);
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 16px;
    min-height: 280px;
  }}

  .panel {{
    display: flex;
    gap: 20px;
    align-items: flex-start;
  }}

  /* ── Avatar card ── */
  .avatar-wrap {{
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }}
  .avatar-ring {{
    width: 120px; height: 120px;
    border-radius: 50%;
    border: 3px solid {p['accent']};
    padding: 3px;
    position: relative;
    animation: ring-pulse 3s ease-in-out infinite;
  }}
  @keyframes ring-pulse {{
    0%,100% {{ box-shadow: 0 0 0 0 {p['accent']}44; }}
    50%      {{ box-shadow: 0 0 0 8px {p['accent']}00; }}
  }}
  .avatar-ring.speaking {{
    animation: ring-speaking 0.4s ease-in-out infinite alternate;
  }}
  @keyframes ring-speaking {{
    from {{ box-shadow: 0 0 0 3px {p['accent']}66; border-color: {p['accent']}; }}
    to   {{ box-shadow: 0 0 0 10px {p['accent']}33; border-color: {p['accent']}cc; }}
  }}
  .avatar-svg {{ border-radius: 50%; display: block; }}

  .name-tag {{
    text-align: center;
    line-height: 1.3;
  }}
  .name-tag .name  {{ color: #f0f0f0; font-size: 13px; font-weight: 600; }}
  .name-tag .title {{
    color: {p['accent']};
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
  }}

  /* Status dot */
  .status-row {{
    display: flex; align-items: center; gap: 5px; margin-top: 2px;
  }}
  .status-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: #6b7280;
  }}
  .status-dot.active {{
    background: {p['accent']};
    animation: dot-blink 1s ease-in-out infinite;
  }}
  @keyframes dot-blink {{
    0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }}
  }}
  .status-text {{ font-size: 10px; color: #9ca3af; }}

  /* ── Speech bubble ── */
  .bubble-wrap {{
    flex: 1;
    position: relative;
    padding-top: 4px;
  }}
  .bubble {{
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 4px 16px 16px 16px;
    padding: 14px 16px;
    color: #e8eaf0;
    font-size: 14px;
    line-height: 1.65;
    position: relative;
    min-height: 80px;
  }}
  .bubble::before {{
    content: '';
    position: absolute;
    left: -10px; top: 18px;
    border: 5px solid transparent;
    border-right-color: rgba(255,255,255,0.12);
  }}

  /* Typing dots — shown while "thinking" */
  .typing-dots {{
    display: none;
    gap: 4px;
    align-items: center;
    padding: 4px 0;
  }}
  .typing-dots.active {{ display: flex; }}
  .typing-dots span {{
    width: 7px; height: 7px; border-radius: 50%;
    background: {p['accent']};
    animation: bounce 1.2s ease-in-out infinite;
  }}
  .typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
  .typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
  @keyframes bounce {{
    0%,80%,100% {{ transform: translateY(0); opacity:0.5; }}
    40%          {{ transform: translateY(-6px); opacity:1; }}
  }}

  /* Speaker wave bars */
  .wave-bars {{
    display: none;
    align-items: flex-end;
    gap: 3px;
    height: 20px;
    margin-top: 10px;
  }}
  .wave-bars.active {{ display: flex; }}
  .wave-bars span {{
    width: 3px;
    background: {p['accent']};
    border-radius: 2px;
    animation: wave 0.8s ease-in-out infinite alternate;
    min-height: 4px;
  }}
  .wave-bars span:nth-child(1) {{ height: 8px;  animation-delay: 0.0s; }}
  .wave-bars span:nth-child(2) {{ height: 16px; animation-delay: 0.1s; }}
  .wave-bars span:nth-child(3) {{ height: 12px; animation-delay: 0.2s; }}
  .wave-bars span:nth-child(4) {{ height: 18px; animation-delay: 0.05s; }}
  .wave-bars span:nth-child(5) {{ height: 10px; animation-delay: 0.15s; }}
  .wave-bars span:nth-child(6) {{ height: 14px; animation-delay: 0.25s; }}
  @keyframes wave {{
    from {{ transform: scaleY(0.4); opacity:0.6; }}
    to   {{ transform: scaleY(1.0); opacity:1.0; }}
  }}

  /* TTS controls */
  .tts-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
    flex-wrap: wrap;
  }}
  .btn {{
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.07);
    color: #e0e0e0;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
  }}
  .btn:hover {{ background: rgba(255,255,255,0.14); }}
  .btn.primary {{
    background: {p['badge']};
    border-color: {p['badge']};
    color: white;
    font-weight: 600;
  }}
  .btn.primary:hover {{ opacity: 0.85; }}
  .btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}

  select {{
    padding: 4px 8px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.07);
    color: #e0e0e0;
    font-size: 11px;
    cursor: pointer;
    font-family: inherit;
  }}
  select option {{ background: #1a1a2e; color: #e0e0e0; }}

  .vol-row {{ display: flex; align-items: center; gap: 6px; }}
  .vol-row label {{ font-size: 11px; color: #9ca3af; }}
  input[type=range] {{
    width: 70px; height: 4px;
    accent-color: {p['accent']};
  }}
</style>
</head>
<body>

<div class="panel">
  <!-- Avatar -->
  <div class="avatar-wrap">
    <div class="avatar-ring" id="avatarRing">
      {_build_avatar_svg(p)}
    </div>
    <div class="name-tag">
      <div class="name">{p['name']}</div>
      <div class="title">{p['title']}</div>
    </div>
    <div class="status-row">
      <div class="status-dot" id="statusDot"></div>
      <div class="status-text" id="statusText">Ready</div>
    </div>
  </div>

  <!-- Speech bubble -->
  <div class="bubble-wrap">
    <div class="bubble" id="bubble">
      <div class="typing-dots" id="typingDots">
        <span></span><span></span><span></span>
      </div>
      <div id="bubbleText">{message}</div>
      <div class="wave-bars" id="waveBars">
        <span></span><span></span><span></span><span></span><span></span><span></span>
      </div>
    </div>

    <div class="tts-bar">
      <button class="btn primary" id="speakBtn" onclick="speakText()">▶ Speak</button>
      <button class="btn" id="stopBtn" onclick="stopSpeech()" disabled>■ Stop</button>
      <select id="voiceSelect" onchange="updateVoice()">
        <option value="">Default voice</option>
      </select>
      <div class="vol-row">
        <label>Vol</label>
        <input type="range" id="volSlider" min="0" max="1" step="0.1" value="0.9">
      </div>
      <div class="vol-row">
        <label>Speed</label>
        <input type="range" id="rateSlider" min="0.6" max="1.4" step="0.05" value="{p['voice_rate']}">
      </div>
    </div>
  </div>
</div>

<script>
const MESSAGE   = `{safe_msg}`;
const PITCH     = {p['voice_pitch']};
const BASE_RATE = {p['voice_rate']};

let synth       = window.speechSynthesis;
let utterance   = null;
let voices      = [];
let chosenVoice = null;

// ── Eye blink animation ────────────────────────────────────────────────────
const leftEye  = document.getElementById('leftEye');
const rightEye = document.getElementById('rightEye');
const mouth    = document.getElementById('mouth');

function blink() {{
  if (!leftEye) return;
  leftEye.setAttribute('ry', '1');
  rightEye.setAttribute('ry', '1');
  setTimeout(() => {{
    leftEye.setAttribute('ry', '6');
    rightEye.setAttribute('ry', '6');
  }}, 120);
}}
setInterval(blink, 3200 + Math.random() * 2000);
setTimeout(blink, 800);

// ── Mouth animation while speaking ────────────────────────────────────────
let mouthAnim = null;
function startMouthAnim() {{
  if (!mouth) return;
  let open = false;
  mouthAnim = setInterval(() => {{
    open = !open;
    if (open) {{
      mouth.setAttribute('d', 'M52 72 Q60 80 68 72');
      mouth.setAttribute('stroke-width', '2.5');
    }} else {{
      mouth.setAttribute('d', 'M52 70 Q60 76 68 70');
      mouth.setAttribute('stroke-width', '2');
    }}
  }}, 180);
}}
function stopMouthAnim() {{
  clearInterval(mouthAnim);
  if (mouth) {{
    mouth.setAttribute('d', 'M52 70 Q60 75 68 70');
    mouth.setAttribute('stroke-width', '2');
  }}
}}

// ── Voice loading ─────────────────────────────────────────────────────────
function loadVoices() {{
  voices = synth.getVoices().filter(v => v.lang.startsWith('en'));
  const sel = document.getElementById('voiceSelect');
  sel.innerHTML = '<option value="">Default voice</option>';
  voices.forEach((v, i) => {{
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = v.name.replace('Microsoft ','').replace(' Online (Natural)','');
    if (v.name.includes('Natural') || v.name.includes('Neural') || v.name.includes('Samantha')) {{
      opt.textContent += ' ✦';
    }}
    sel.appendChild(opt);
  }});
  // Auto-select best natural/neural voice available
  for (let i = 0; i < voices.length; i++) {{
    if (voices[i].name.includes('Natural') || voices[i].name.includes('Neural')) {{
      sel.value = i;
      chosenVoice = voices[i];
      break;
    }}
  }}
}}

if (synth.onvoiceschanged !== undefined) {{
  synth.onvoiceschanged = loadVoices;
}}
setTimeout(loadVoices, 300);

function updateVoice() {{
  const idx = document.getElementById('voiceSelect').value;
  chosenVoice = idx !== '' ? voices[parseInt(idx)] : null;
}}

// ── TTS ───────────────────────────────────────────────────────────────────
function setSpeakingState(speaking) {{
  const ring   = document.getElementById('avatarRing');
  const dot    = document.getElementById('statusDot');
  const status = document.getElementById('statusText');
  const wave   = document.getElementById('waveBars');
  const speakBtn = document.getElementById('speakBtn');
  const stopBtn  = document.getElementById('stopBtn');

  if (speaking) {{
    ring.classList.add('speaking');
    dot.classList.add('active');
    status.textContent = 'Speaking…';
    wave.classList.add('active');
    speakBtn.disabled = true;
    stopBtn.disabled  = false;
    startMouthAnim();
  }} else {{
    ring.classList.remove('speaking');
    dot.classList.remove('active');
    status.textContent = 'Ready';
    wave.classList.remove('active');
    speakBtn.disabled = false;
    stopBtn.disabled  = true;
    stopMouthAnim();
  }}
}}

function speakText() {{
  if (synth.speaking) synth.cancel();
  if (!MESSAGE.trim()) return;

  utterance = new SpeechSynthesisUtterance(MESSAGE);
  utterance.pitch  = PITCH;
  utterance.rate   = parseFloat(document.getElementById('rateSlider').value);
  utterance.volume = parseFloat(document.getElementById('volSlider').value);
  if (chosenVoice) utterance.voice = chosenVoice;

  utterance.onstart = () => setSpeakingState(true);
  utterance.onend   = () => setSpeakingState(false);
  utterance.onerror = () => setSpeakingState(false);

  synth.speak(utterance);
}}

function stopSpeech() {{
  synth.cancel();
  setSpeakingState(false);
}}

// Auto-speak on load
window.addEventListener('load', () => {{
  setTimeout(() => {{
    if (MESSAGE.trim()) speakText();
  }}, 600);
}});
</script>
</body>
</html>
"""
    return html


def _build_avatar_svg(p: dict) -> str:
    """Build a distinct illustrated SVG face for each persona."""
    skin   = p["skin"]
    hair   = p["hair"]
    shirt  = p["shirt"]
    accent = p["accent"]

    return f"""<svg class="avatar-svg" width="112" height="112" viewBox="0 0 112 112"
         xmlns="http://www.w3.org/2000/svg">
  <!-- Background circle -->
  <circle cx="56" cy="56" r="56" fill="{p['bg_to']}"/>

  <!-- Shirt / shoulders -->
  <ellipse cx="56" cy="108" rx="38" ry="22" fill="{shirt}"/>
  <rect x="22" y="88" width="68" height="30" fill="{shirt}" rx="4"/>

  <!-- Collar detail -->
  <polygon points="56,88 48,102 56,96" fill="white" opacity="0.25"/>
  <polygon points="56,88 64,102 56,96" fill="white" opacity="0.25"/>

  <!-- Neck -->
  <rect x="48" y="78" width="16" height="16" rx="6" fill="{skin}"/>

  <!-- Head -->
  <ellipse cx="56" cy="54" rx="26" ry="28" fill="{skin}"/>

  <!-- Hair -->
  <ellipse cx="56" cy="30" rx="26" ry="12" fill="{hair}"/>
  <rect x="30" y="28" width="52" height="16" fill="{hair}" rx="2"/>

  <!-- Ears -->
  <ellipse cx="30" cy="54" rx="5" ry="7" fill="{skin}"/>
  <ellipse cx="82" cy="54" rx="5" ry="7" fill="{skin}"/>
  <ellipse cx="30" cy="54" rx="3" ry="5" fill="{p['bg_to']}" opacity="0.3"/>
  <ellipse cx="82" cy="54" rx="3" ry="5" fill="{p['bg_to']}" opacity="0.3"/>

  <!-- Eyebrows -->
  <path d="M42 42 Q48 39 53 41" stroke="{hair}" stroke-width="2" fill="none" stroke-linecap="round"/>
  <path d="M59 41 Q64 39 70 42" stroke="{hair}" stroke-width="2" fill="none" stroke-linecap="round"/>

  <!-- Eyes with animation targets -->
  <ellipse id="leftEye"  cx="47" cy="52" rx="5.5" ry="6" fill="white"/>
  <ellipse id="rightEye" cx="65" cy="52" rx="5.5" ry="6" fill="white"/>
  <!-- Irises -->
  <circle cx="47" cy="53" r="3.5" fill="{accent}" opacity="0.9"/>
  <circle cx="65" cy="53" r="3.5" fill="{accent}" opacity="0.9"/>
  <!-- Pupils -->
  <circle cx="47" cy="53" r="2" fill="#1a1a2e"/>
  <circle cx="65" cy="53" r="2" fill="#1a1a2e"/>
  <!-- Eye shine -->
  <circle cx="48.5" cy="51.5" r="1" fill="white" opacity="0.8"/>
  <circle cx="66.5" cy="51.5" r="1" fill="white" opacity="0.8"/>

  <!-- Nose -->
  <path d="M54 57 Q56 63 58 57" stroke="{skin}" stroke-width="1.5"
        fill="none" stroke-linecap="round" opacity="0.6"/>
  <ellipse cx="53" cy="62" rx="2" ry="1.5" fill="{skin}" opacity="0.5"/>
  <ellipse cx="59" cy="62" rx="2" ry="1.5" fill="{skin}" opacity="0.5"/>

  <!-- Mouth (animated via JS) -->
  <path id="mouth" d="M52 70 Q60 75 68 70" stroke="{hair}" stroke-width="2"
        fill="none" stroke-linecap="round"/>

  <!-- Blush marks -->
  <ellipse cx="38" cy="62" rx="5" ry="3" fill="#ff9999" opacity="0.2"/>
  <ellipse cx="74" cy="62" rx="5" ry="3" fill="#ff9999" opacity="0.2"/>
</svg>"""


# ── Convenience wrapper ────────────────────────────────────────────────────────
def avatar_panel(persona: str, message: str, height: int = 280) -> None:
    """Call this from app.py to render the avatar into the Streamlit page."""
    import streamlit.components.v1 as components
    html = get_avatar_html(persona, message)
    components.html(html, height=height, scrolling=False)
