from typing import Optional

try:
    import folium
    from folium.plugins import MeasureControl, Fullscreen
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


class GeoTagger:
    """Manages geo-tagging metadata for agricultural image analysis."""

    def __init__(self):
        self.locations: list = []

    def add(self, lat: float, lon: float, name: str, metadata: Optional[dict] = None):
        self.locations.append({"lat": lat, "lon": lon, "name": name, "metadata": metadata or {}})

    def context_string(self, lat: float, lon: float, metadata: Optional[dict] = None) -> str:
        parts = [f"Location: ({lat:.4f}, {lon:.4f})"]
        if metadata:
            for key in ("soil_type", "crop_type", "season", "rainfall_mm", "temp_c"):
                if key in metadata:
                    parts.append(f"{key.replace('_', ' ').title()}: {metadata[key]}")
        return " | ".join(parts)


def create_location_map(
    lat: float,
    lon: float,
    name: str,
    zoom: int = 14,
    analysis_results: Optional[dict] = None,
) -> "folium.Map":
    """Build a folium map with satellite imagery and an analysis marker."""
    if not FOLIUM_AVAILABLE:
        raise ImportError("folium is not installed. Run: pip install folium streamlit-folium")

    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap")

    # Satellite tile layer
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # NDVI-style green tile (OpenTopoMap as terrain option)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Terrain",
        overlay=False,
        control=True,
    ).add_to(m)

    # Build popup content
    popup_html = f"<b>{name}</b><br>Lat: {lat:.5f}<br>Lon: {lon:.5f}"
    if analysis_results:
        score = analysis_results.get("health_score", "N/A")
        alert = "YES" if analysis_results.get("alert_required") else "NO"
        popup_html += f"<br>Health Score: {score}/100<br>Alert: {alert}"

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=name,
        icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
    ).add_to(m)

    folium.Circle(
        location=[lat, lon],
        radius=500,
        color="green",
        fill=True,
        fill_opacity=0.08,
        popup="Analysis Area (~500 m radius)",
    ).add_to(m)

    MeasureControl(position="topright").add_to(m)
    Fullscreen(position="topright").add_to(m)
    folium.LayerControl().add_to(m)

    return m


def add_anomaly_markers(m: "folium.Map", anomalies: list, lat: float, lon: float):
    """Overlay coloured anomaly markers near the field center."""
    if not FOLIUM_AVAILABLE or not anomalies:
        return m

    severity_colors = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "beige", "LOW": "green"}
    import math

    for i, anomaly in enumerate(anomalies):
        angle = (360 / max(len(anomalies), 1)) * i
        offset = 0.003
        a_lat = lat + offset * math.cos(math.radians(angle))
        a_lon = lon + offset * math.sin(math.radians(angle))
        sev = anomaly.get("severity", "LOW").upper()
        color = severity_colors.get(sev, "gray")

        folium.CircleMarker(
            location=[a_lat, a_lon],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(anomaly.get("description", ""), max_width=200),
            tooltip=f"[{sev}] anomaly",
        ).add_to(m)

    return m
