import io
import os
from datetime import datetime

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.alert-critical{background:#c0392b;color:#fff;padding:10px 14px;border-radius:6px;margin:4px 0;font-weight:600}
.alert-high    {background:#e67e22;color:#fff;padding:10px 14px;border-radius:6px;margin:4px 0;font-weight:600}
.alert-medium  {background:#f1c40f;color:#222;padding:10px 14px;border-radius:6px;margin:4px 0;font-weight:600}
.alert-low     {background:#27ae60;color:#fff;padding:10px 14px;border-radius:6px;margin:4px 0;font-weight:600}
.health-score  {font-size:2.4rem;font-weight:700;text-align:center;padding:12px 0}
.agent-badge   {font-size:0.75rem;font-weight:600;background:#1a1a2e;color:#e2e8f0;
                padding:3px 10px;border-radius:12px;display:inline-block;margin-bottom:6px}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("current_image", None),
    ("geo_context", None),
    ("terrain_fig", None),
    ("terrain_stats", None),
    ("gee_results", None),
    ("monitoring_queue", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ────────────────────────────────────────────────────────────────────
def to_jpeg_bytes(uploaded) -> bytes:
    img = Image.open(uploaded)
    if img.mode not in ("RGB",):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def health_gauge(score: int):
    color = (
        "#27ae60" if score >= 75 else "#f1c40f" if score >= 50 else "#e67e22" if score >= 25 else "#c0392b"
    )
    st.markdown(
        f'<div class="health-score" style="color:{color}">Health Score: {score}/100</div>',
        unsafe_allow_html=True,
    )


def anomaly_cards(anomalies: list):
    if not anomalies:
        st.success("No anomalies detected in this image.")
        return
    icons = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "ℹ️"}
    for a in anomalies:
        sev = a.get("severity", "LOW").upper()
        icon = icons.get(sev, "ℹ️")
        st.markdown(
            f'<div class="alert-{sev.lower()}">{icon} [{sev}] {a["description"]}</div>',
            unsafe_allow_html=True,
        )


def api_key_ok() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌾 AgriVision AI")
    st.caption("Multimodal Farm Intelligence Platform")
    st.divider()

    st.subheader("API Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    model = st.selectbox(
        "Claude Model",
        ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"],
        index=0,
    )

    st.divider()
    st.subheader("Google Earth Engine")
    gee_project = st.text_input("GEE Project ID", placeholder="your-gee-project")
    gee_sa = st.text_input("Service Account", placeholder="sa@project.iam.gserviceaccount.com")

    st.divider()
    st.subheader("Explanation Audience")
    audience = st.selectbox("Explain results for:", ["farmer", "agronomist", "investor", "researcher"])

    st.divider()
    if st.session_state.history:
        st.subheader(f"History ({len(st.session_state.history)})")
        for entry in reversed(st.session_state.history[-5:]):
            with st.expander(f"🕐 {entry['ts']}  {entry['name'][:18]}"):
                if "score" in entry:
                    st.metric("Health", f"{entry['score']}/100")
                if entry.get("alerts"):
                    st.caption(f"{entry['alerts']} anomaly alert(s)")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.caption("Powered by Claude AI + Google Earth Engine")

# ── Main header ────────────────────────────────────────────────────────────────
st.title("🌾 AgriVision AI — Multimodal Farm Intelligence")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📸 Image Analysis",
        "🗺️ Geo-Tagging",
        "📡 Real-Time Monitoring",
        "🏔️ 3D Terrain",
        "🔮 What-If Scenarios",
        "🛰️ Earth Engine",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Multi-Agent Image Analysis")
    st.caption("Upload one or more farm / satellite images for layered AI analysis.")

    left, right = st.columns([1, 1])

    with left:
        files = st.file_uploader(
            "Upload Farm Images",
            type=["jpg", "jpeg", "png", "tiff", "tif"],
            accept_multiple_files=True,
        )

        if files:
            idx = 0
            if len(files) > 1:
                idx = st.selectbox(
                    "Select image",
                    range(len(files)),
                    format_func=lambda i: files[i].name,
                )
            chosen = files[idx]
            img_bytes = to_jpeg_bytes(chosen)
            st.session_state.current_image = img_bytes
            img_pil = Image.open(io.BytesIO(img_bytes))
            st.image(img_pil, caption=chosen.name, use_container_width=True)
            st.caption(f"{img_pil.size[0]}×{img_pil.size[1]} px · {len(img_bytes)//1024} KB")

    with right:
        if not st.session_state.current_image:
            st.info("Upload an image to begin.")
            st.markdown(
                """
**Multi-agent pipeline:**
1. **Interpreter Agent** — deep technical analysis
2. **Explainer Agent** — plain-language translation
3. **Anomaly Detector** — severity-coded alerts
4. Results are pinned to GPS via the Geo-Tagging tab
"""
            )
        else:
            run_interp = st.checkbox("Interpreter Agent", value=True)
            run_expl = st.checkbox("Explainer Agent", value=True)
            run_anom = st.checkbox("Anomaly Detector", value=True)

            go_btn = st.button("🔍 Run Multi-Agent Analysis", type="primary", use_container_width=True)

            if go_btn:
                if not api_key_ok():
                    st.error("Enter your Anthropic API key in the sidebar first.")
                else:
                    from agents import AnomalyDetectorAgent, ExplainerAgent, InterpreterAgent

                    results: dict = {}

                    if run_interp:
                        with st.spinner("Interpreter Agent…"):
                            try:
                                agent = InterpreterAgent(model=model)
                                results["interp"] = agent.analyze(
                                    st.session_state.current_image,
                                    st.session_state.geo_context,
                                )
                            except Exception as e:
                                st.error(f"Interpreter error: {e}")

                    if run_expl and "interp" in results:
                        with st.spinner("Explainer Agent…"):
                            try:
                                agent = ExplainerAgent(model=model)
                                results["expl"] = agent.explain(
                                    results["interp"]["raw_analysis"], audience=audience
                                )
                            except Exception as e:
                                st.error(f"Explainer error: {e}")

                    if run_anom:
                        with st.spinner("Anomaly Detector…"):
                            try:
                                agent = AnomalyDetectorAgent(model=model)
                                results["anom"] = agent.detect(st.session_state.current_image)
                            except Exception as e:
                                st.error(f"Anomaly Detector error: {e}")

                    # Persist to history
                    hist_entry = {
                        "ts": datetime.now().strftime("%H:%M:%S"),
                        "name": chosen.name,
                    }
                    if "anom" in results:
                        hist_entry["score"] = results["anom"]["health_score"]
                        hist_entry["alerts"] = int(results["anom"]["alert_required"])
                    st.session_state.history.append(hist_entry)

                    st.divider()

                    # Health gauge
                    if "anom" in results:
                        health_gauge(results["anom"]["health_score"])
                        if results["anom"]["alert_required"]:
                            st.error("🚨 ALERT: Immediate attention required!")

                    if "interp" in results:
                        with st.expander("📊 Interpreter Agent — Technical Analysis", expanded=True):
                            st.markdown(
                                '<span class="agent-badge">🤖 Interpreter Agent</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(results["interp"]["raw_analysis"])
                            anomaly_cards(results["interp"]["anomalies"])

                    if "expl" in results:
                        with st.expander("💬 Explainer Agent — Plain Language", expanded=True):
                            st.markdown(
                                '<span class="agent-badge">🤖 Explainer Agent</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(results["expl"])

                    if "anom" in results:
                        with st.expander("🔴 Anomaly Detector Report", expanded=True):
                            st.markdown(
                                '<span class="agent-badge">🤖 Anomaly Detector Agent</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(results["anom"]["full_report"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEO-TAGGING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Geo-Tagging & Environmental Context")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Location Input")
        method = st.radio("Input method", ["Manual Coordinates", "Named Location (demo)"])

        if method == "Manual Coordinates":
            lat = st.number_input("Latitude", value=37.7749, format="%.6f", min_value=-90.0, max_value=90.0)
            lon = st.number_input("Longitude", value=-122.4194, format="%.6f", min_value=-180.0, max_value=180.0)
            region = st.text_input("Field / Region Name", value="Sample Field")
        else:
            presets = {
                "Central Valley, CA": (36.7783, -119.4179),
                "Iowa Corn Belt": (41.8780, -93.0977),
                "Kansas Wheat Plains": (38.5267, -98.2870),
                "Mississippi Delta": (33.1387, -90.1862),
            }
            choice = st.selectbox("Select preset location", list(presets.keys()))
            lat, lon = presets[choice]
            region = choice

        st.subheader("Environmental Context")
        soil = st.selectbox("Soil Type", ["Loam", "Clay", "Sandy", "Silt", "Clay-Loam", "Unknown"])
        crop = st.selectbox("Crop Type", ["Corn", "Wheat", "Soybeans", "Cotton", "Rice", "Alfalfa", "Unknown"])
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        rain_mm = st.slider("Recent Rainfall (mm/week)", 0, 200, 25)
        temp_c = st.slider("Average Temperature (°C)", -10, 50, 22)

        if st.button("📌 Set Geo-Context", type="primary"):
            st.session_state.geo_context = {
                "lat": lat, "lon": lon, "region": region,
                "soil_type": soil, "crop_type": crop,
                "season": season, "rainfall_mm": rain_mm, "temp_c": temp_c,
            }
            st.success(f"Context saved: {region} ({lat:.4f}, {lon:.4f})")

    with right:
        st.subheader("Interactive Map")
        display_lat = st.session_state.geo_context["lat"] if st.session_state.geo_context else lat
        display_lon = st.session_state.geo_context["lon"] if st.session_state.geo_context else lon
        display_name = st.session_state.geo_context["region"] if st.session_state.geo_context else region

        try:
            from geo_utils import add_anomaly_markers, create_location_map
            from streamlit_folium import st_folium

            anomaly_list = []
            if st.session_state.history:
                last = st.session_state.history[-1]
                # Pass anomalies if available from last run (simplified)

            m = create_location_map(display_lat, display_lon, display_name)
            st_folium(m, width=None, height=420, returned_objects=[])
        except ImportError:
            st.warning("Install `streamlit-folium` and `folium` for interactive maps.")
            st.map({"lat": [display_lat], "lon": [display_lon]})

        if st.session_state.geo_context:
            ctx = st.session_state.geo_context
            st.subheader("Saved Context")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Lat", f"{ctx['lat']:.4f}")
                st.metric("Soil", ctx["soil_type"])
                st.metric("Rainfall", f"{ctx['rainfall_mm']} mm/wk")
            with c2:
                st.metric("Lon", f"{ctx['lon']:.4f}")
                st.metric("Crop", ctx["crop_type"])
                st.metric("Temperature", f"{ctx['temp_c']}°C")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REAL-TIME MONITORING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Real-Time Monitoring & Auto-Flag System")
    st.caption("Queue multiple images for sequential anomaly scanning with severity alerts.")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Monitoring Queue")
        mon_files = st.file_uploader(
            "Add images to queue",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="mon_upload",
        )
        if mon_files:
            if st.button("➕ Add to Queue"):
                for f in mon_files:
                    st.session_state.monitoring_queue.append(
                        {
                            "name": f.name,
                            "data": to_jpeg_bytes(f),
                            "ts": datetime.now().strftime("%H:%M:%S"),
                            "status": "pending",
                            "result": None,
                        }
                    )
                st.success(f"Added {len(mon_files)} image(s).")

        status_icons = {"pending": "⏳", "analyzing": "🔄", "ok": "✅", "alert": "🚨"}
        for item in st.session_state.monitoring_queue:
            icon = status_icons.get(item["status"], "⏳")
            st.text(f"{icon}  {item['name']}  ({item['ts']})")

        c1, c2 = st.columns(2)
        with c1:
            run_mon = st.button(
                "▶ Run Monitoring",
                type="primary",
                disabled=not st.session_state.monitoring_queue,
            )
        with c2:
            if st.button("🗑 Clear Queue"):
                st.session_state.monitoring_queue = []
                st.rerun()

        alert_level = st.selectbox(
            "Minimum alert severity to flag",
            ["ALL", "MEDIUM+", "HIGH+", "CRITICAL only"],
        )

    with right:
        st.subheader("Live Analysis Feed")

        if run_mon:
            if not api_key_ok():
                st.error("API key required.")
            else:
                from agents import AnomalyDetectorAgent

                agent = AnomalyDetectorAgent(model=model)
                pending = [i for i in st.session_state.monitoring_queue if i["status"] == "pending"]

                for item in pending:
                    item["status"] = "analyzing"
                    prog = st.progress(0, text=f"Scanning {item['name']}…")
                    try:
                        prog.progress(40)
                        result = agent.detect(item["data"])
                        prog.progress(100, text=f"✅ Done — Score {result['health_score']}/100")
                        item["result"] = result
                        item["status"] = "alert" if result["alert_required"] else "ok"
                    except Exception as e:
                        st.error(f"Error on {item['name']}: {e}")
                        item["status"] = "ok"

        # Display completed results
        done = [i for i in st.session_state.monitoring_queue if i["status"] in ("ok", "alert")]
        if done:
            for item in done:
                r = item.get("result")
                if not r:
                    continue
                score = r["health_score"]
                label = "🚨" if item["status"] == "alert" else "✅"
                with st.expander(f"{label} {item['name']} — Health {score}/100"):
                    health_gauge(score)
                    st.progress(score / 100)
                    if r["alert_required"]:
                        st.error("🚨 Immediate attention required!")
                    st.markdown(r["full_report"])
        elif not st.session_state.monitoring_queue:
            st.info("Add images to the queue and click Run Monitoring.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 3D TERRAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("3D Terrain Intelligence & Depth Perception")
    st.caption(
        "Generates AI-derived depth maps from image colour/vegetation patterns. "
        "Upload LiDAR CSV for real terrain data."
    )

    left, right = st.columns([1, 1])

    with left:
        terrain_file = st.file_uploader(
            "Upload terrain / field image", type=["jpg", "jpeg", "png"], key="terrain_up"
        )

        source_img: bytes | None = None
        if terrain_file:
            source_img = to_jpeg_bytes(terrain_file)
            st.image(Image.open(io.BytesIO(source_img)), caption="Source", use_container_width=True)
        elif st.session_state.current_image:
            source_img = st.session_state.current_image
            st.image(
                Image.open(io.BytesIO(source_img)),
                caption="Using Analysis tab image",
                use_container_width=True,
            )
        else:
            st.info("Upload an image here or load one in the Analysis tab.")

        st.subheader("Visualisation Options")
        resolution = st.slider("3D Resolution (grid points)", 20, 120, 60)
        viz_type = st.selectbox("Visualisation Type", ["Surface Plot", "Contour Map", "Wireframe", "3D Scatter"])
        colorscale = st.selectbox(
            "Colour Scheme",
            ["Viridis", "YlOrRd", "Blues", "Earth", "Plasma"],
        )

        # Optional real LiDAR upload
        st.subheader("LiDAR Data (optional)")
        lidar_file = st.file_uploader(
            "Upload LiDAR CSV (columns: x, y, z)", type=["csv", "txt"], key="lidar_up"
        )

        gen_btn = st.button("Generate 3D Terrain", type="primary", disabled=source_img is None)

        if gen_btn and source_img:
            with st.spinner("Building 3D model…"):
                from depth_analyzer import DepthAnalyzer

                da = DepthAnalyzer()

                if lidar_file is not None:
                    # Real LiDAR path
                    import csv

                    rows_data = list(csv.reader(io.StringIO(lidar_file.getvalue().decode())))
                    try:
                        pts = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows_data[1:]])
                        side = int(np.sqrt(len(pts)))
                        depth_arr = pts[:side**2, 2].reshape(side, side)
                        st.success("LiDAR data loaded — using real elevation values.")
                    except Exception:
                        st.warning("Could not parse LiDAR file; falling back to image depth.")
                        depth_arr = da.generate_depth_map(Image.open(io.BytesIO(source_img)), resolution)
                else:
                    depth_arr = da.generate_depth_map(Image.open(io.BytesIO(source_img)), resolution)

                st.session_state.terrain_stats = da.terrain_stats(depth_arr)

                if viz_type == "Surface Plot":
                    st.session_state.terrain_fig = da.create_surface_plot(depth_arr, colorscale)
                elif viz_type == "Contour Map":
                    st.session_state.terrain_fig = da.create_contour_map(depth_arr, colorscale)
                elif viz_type == "Wireframe":
                    st.session_state.terrain_fig = da.create_wireframe(depth_arr)
                else:
                    st.session_state.terrain_fig = da.create_scatter_3d(depth_arr, colorscale)

    with right:
        st.subheader("3D View")
        if st.session_state.terrain_fig:
            st.plotly_chart(st.session_state.terrain_fig, use_container_width=True)

            if st.session_state.terrain_stats:
                stats = st.session_state.terrain_stats
                st.subheader("Terrain Statistics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Min Elevation", f"{stats['min_elevation']} m")
                c2.metric("Max Elevation", f"{stats['max_elevation']} m")
                c3.metric("Mean Elevation", f"{stats['mean_elevation']} m")
                c1.metric("Elevation Range", f"{stats['elevation_range']} m")
                c2.metric("Mean Slope", f"{stats['mean_slope']}°")
                c3.metric("High-Slope Area", f"{stats['high_slope_pct']}%")
        else:
            st.info("Configure options on the left and click **Generate 3D Terrain**.")
            st.markdown(
                """
**3D Terrain Intelligence features:**
- Pseudo-depth estimation from vegetation density & luminance
- Surface / contour / wireframe / point-cloud views
- Slope analysis for drainage and erosion risk assessment
- Real LiDAR point-cloud support via CSV upload
- Colour schemes tuned for vegetation, moisture, or heat stress
"""
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WHAT-IF SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Visual 'What-If' Scenario Analysis")
    st.caption("Ask the AI hypothetical questions grounded in what it sees in the image.")

    left, right = st.columns([1, 1])

    with left:
        if st.session_state.current_image:
            st.image(
                Image.open(io.BytesIO(st.session_state.current_image)),
                caption="Active analysis image",
                use_container_width=True,
            )
        else:
            wi_file = st.file_uploader("Upload image for scenario analysis", type=["jpg", "jpeg", "png"], key="wi_up")
            if wi_file:
                st.session_state.current_image = to_jpeg_bytes(wi_file)
                st.image(Image.open(io.BytesIO(st.session_state.current_image)), use_container_width=True)

        st.subheader("Scenario")
        presets = [
            "What would happen if rainfall doubled this month?",
            "What if irrigation stopped for 2 weeks?",
            "What if temperatures rose 5 °C above normal?",
            "What if a fungal blight spread from the stressed areas?",
            "What if we switched this field to organic management?",
            "What would a drought year look like for this crop?",
            "What if nitrogen fertiliser was applied at double rate?",
            "Custom question…",
        ]
        preset = st.selectbox("Quick scenarios", presets)
        question = (
            st.text_area("Your what-if question", placeholder="What would happen if…", height=90)
            if preset == "Custom question…"
            else preset
        )

        with st.expander("Add Context (optional)"):
            weather = st.text_input("Weather forecast", placeholder="e.g., drought expected next 30 days")
            market = st.text_input("Market context", placeholder="e.g., corn futures at 5-year high")
            planned = st.text_input("Planned intervention", placeholder="e.g., adding drip irrigation")

        extra_ctx: dict = {}
        if weather:
            extra_ctx["weather_forecast"] = weather
        if market:
            extra_ctx["market_context"] = market
        if planned:
            extra_ctx["planned_intervention"] = planned
        if st.session_state.geo_context:
            extra_ctx["location"] = st.session_state.geo_context

        stream_on = st.toggle("Stream response in real-time", value=True)
        ask_btn = st.button(
            "🔮 Analyse Scenario",
            type="primary",
            disabled=not (st.session_state.current_image and question),
        )

    with right:
        st.subheader("Scenario Result")
        if ask_btn:
            if not api_key_ok():
                st.error("Enter your Anthropic API key in the sidebar.")
            elif not st.session_state.current_image:
                st.warning("Upload an image first.")
            else:
                from agents import WhatIfAgent

                agent = WhatIfAgent(model=model)
                st.markdown(f"**Scenario:** _{question}_")
                st.divider()

                if stream_on:
                    box = st.empty()
                    full = ""
                    with st.spinner("Analysing…"):
                        for chunk in agent.stream(
                            st.session_state.current_image, question
                        ):
                            full += chunk
                            box.markdown(full)
                else:
                    with st.spinner("Analysing…"):
                        result = agent.answer(
                            st.session_state.current_image,
                            question,
                            extra_ctx or None,
                        )
                        st.markdown(result)
        else:
            st.info("Select a scenario and click **Analyse Scenario**.")
            st.markdown(
                """
**Example questions:**
- 🌧️ Rainfall, flood, or drought projections
- 🌡️ Heat stress thresholds for current crop stage
- 💧 Impact of irrigation changes
- 🐛 Pest or disease spread scenarios
- 🌱 Crop rotation economic trade-offs
- 📈 Yield projections under different management
"""
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — GOOGLE EARTH ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Google Earth Engine — Real-Time Satellite Data")
    st.caption("Fetch NDVI, NDWI, and other vegetation indices from live satellite imagery.")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Query Configuration")

        if st.session_state.geo_context:
            ctx = st.session_state.geo_context
            gee_lat, gee_lon = ctx["lat"], ctx["lon"]
            st.info(f"Using saved location: **{ctx['region']}** ({gee_lat:.4f}, {gee_lon:.4f})")
        else:
            gee_lat = st.number_input("Latitude", value=37.7749, format="%.6f", key="g_lat")
            gee_lon = st.number_input("Longitude", value=-122.4194, format="%.6f", key="g_lon")

        d1, d2 = st.columns(2)
        with d1:
            start = st.date_input("Start Date", value=datetime(2024, 1, 1))
        with d2:
            end = st.date_input("End Date", value=datetime.now())

        satellite = st.selectbox(
            "Satellite",
            ["Sentinel-2 (10 m)", "Landsat 8 (30 m)", "MODIS (250 m)"],
        )
        sat_key = satellite.split()[0].lower().replace("-", "")
        if sat_key not in ("sentinel2", "landsat", "modis"):
            sat_key = "sentinel2"

        indices = st.multiselect(
            "Vegetation Indices to fetch",
            ["NDVI", "NDWI", "EVI", "SAVI", "LST"],
            default=["NDVI", "NDWI"],
        )

        fetch_btn = st.button("🛰️ Fetch Satellite Data", type="primary")

        if fetch_btn:
            from gee_connector import GEEConnector

            gee = GEEConnector()
            with st.spinner("Connecting to Google Earth Engine…"):
                if gee_project:
                    ok = gee.connect(gee_project, gee_sa or None)
                    if not ok:
                        st.error("GEE connection failed. Check your project ID / service account.")
                        st.stop()
                    label = "live"
                else:
                    st.warning("No GEE project configured — displaying demo data.")
                    label = "demo"

            with st.spinner("Fetching satellite imagery…"):
                results = gee.get_vegetation_indices(
                    gee_lat, gee_lon, str(start), str(end), sat_key
                )
                results["_label"] = label
                st.session_state.gee_results = results
                st.success(f"Data fetched ({label} mode).")

        st.divider()
        st.markdown(
            """
**GEE Setup:**
1. Create account at [earthengine.google.com](https://earthengine.google.com)
2. Enable Earth Engine API in Google Cloud Console
3. Create a Service Account with *Editor* access
4. Download the key JSON and enter the email above
5. Enter your Cloud Project ID

**Supported indices:**
| Index | Measures |
|-------|----------|
| NDVI | Vegetation density & health |
| NDWI | Canopy water content |
| EVI | Enhanced vegetation (less saturation) |
| SAVI | Soil-adjusted vegetation |
| LST | Land Surface Temperature |
"""
        )

    with right:
        st.subheader("Satellite Analysis Results")
        results = st.session_state.gee_results

        if results:
            import plotly.graph_objects as go

            ndvi = results.get("ndvi")
            ndwi = results.get("ndwi")
            src = results.get("source", "unknown")
            label = results.get("_label", "")

            c1, c2, c3 = st.columns(3)
            if ndvi is not None:
                baseline = 0.5
                c1.metric("NDVI", f"{ndvi:.3f}", delta=f"{ndvi - baseline:+.3f} vs baseline")
            if ndwi is not None:
                c2.metric("NDWI", f"{ndwi:.3f}")
            c3.metric("Source", src.title() + (f" [{label}]" if label else ""))

            # NDVI interpretation band
            if ndvi is not None:
                if ndvi > 0.6:
                    st.success("🌿 Healthy, dense vegetation")
                elif ndvi > 0.4:
                    st.info("🌱 Moderate vegetation — monitor for stress")
                elif ndvi > 0.2:
                    st.warning("⚠️ Below-average vegetation health")
                else:
                    st.error("🚨 Critically low vegetation — immediate review needed")

            # Time-series chart
            ts = results.get("time_series", {})
            if ts.get("dates"):
                fig = go.Figure()
                if ts.get("ndvi_values"):
                    fig.add_trace(
                        go.Scatter(
                            x=ts["dates"],
                            y=ts["ndvi_values"],
                            mode="lines+markers",
                            name="NDVI",
                            line=dict(color="#27ae60", width=2),
                            marker=dict(size=5),
                        )
                    )
                if ts.get("ndwi_values"):
                    fig.add_trace(
                        go.Scatter(
                            x=ts["dates"],
                            y=ts["ndwi_values"],
                            mode="lines+markers",
                            name="NDWI",
                            line=dict(color="#2980b9", width=2, dash="dot"),
                            marker=dict(size=5),
                        )
                    )
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Healthy baseline")
                fig.update_layout(
                    title="Vegetation Index Time Series",
                    xaxis_title="Date",
                    yaxis_title="Index Value",
                    hovermode="x unified",
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Satellite anomaly alerts
            if results.get("anomalies"):
                st.subheader("Satellite-Detected Anomalies")
                anomaly_cards(results["anomalies"])
        else:
            st.info("Configure the query and click **Fetch Satellite Data**.")
