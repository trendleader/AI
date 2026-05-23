import numpy as np
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import plotly.graph_objects as go


class DepthAnalyzer:
    """Generates pseudo-depth maps and 3D terrain visualizations from agricultural imagery."""

    def generate_depth_map(self, image: Image.Image, resolution: int = 50) -> np.ndarray:
        """
        Derive a pseudo-depth array from image intensity patterns.
        Uses a vegetation-weighted blend of NDVI proxy and luminance as the
        elevation surrogate — brighter/greener areas map to higher 'elevation'.
        """
        img = image.resize((resolution, resolution), Image.LANCZOS).convert("RGB")
        arr = np.array(img, dtype=float)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # NDVI-like proxy: emphasises healthy green cover
        ndvi_proxy = (g - r) / (g + r + 1e-8)

        # Luminance channel for topographic texture
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

        depth = 0.65 * ndvi_proxy + 0.35 * lum

        # Normalise to 0–100 m
        lo, hi = depth.min(), depth.max()
        depth = (depth - lo) / (hi - lo + 1e-8) * 100.0

        if SCIPY_AVAILABLE:
            depth = gaussian_filter(depth, sigma=1.5)

        return depth

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _base_layout(self, title: str) -> dict:
        return dict(
            title=title,
            margin=dict(l=0, r=0, t=40, b=0),
        )

    def create_surface_plot(self, depth: np.ndarray, colorscale: str = "Viridis") -> go.Figure:
        fig = go.Figure(
            data=[
                go.Surface(
                    z=depth,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title="Elevation (m)"),
                    contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                )
            ]
        )
        fig.update_layout(
            **self._base_layout("3D Terrain Surface — AI-Derived Depth Map"),
            scene=dict(
                xaxis_title="East–West",
                yaxis_title="North–South",
                zaxis_title="Elevation (m)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
        )
        return fig

    def create_contour_map(self, depth: np.ndarray, colorscale: str = "Viridis") -> go.Figure:
        fig = go.Figure(
            data=[
                go.Contour(
                    z=depth,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title="Elevation (m)"),
                    contours=dict(
                        coloring="heatmap",
                        showlines=True,
                        showlabels=True,
                        labelfont=dict(size=9, color="white"),
                    ),
                    line_smoothing=0.8,
                )
            ]
        )
        fig.update_layout(
            **self._base_layout("Terrain Contour Map"),
            xaxis_title="East–West",
            yaxis_title="North–South",
        )
        return fig

    def create_wireframe(self, depth: np.ndarray) -> go.Figure:
        rows, cols = depth.shape
        step = max(1, rows // 20)
        fig = go.Figure()

        for i in range(0, rows, step):
            fig.add_trace(
                go.Scatter3d(
                    x=np.arange(cols),
                    y=np.full(cols, i),
                    z=depth[i, :],
                    mode="lines",
                    line=dict(color="#00ff88", width=1),
                    showlegend=False,
                )
            )
        for j in range(0, cols, step):
            fig.add_trace(
                go.Scatter3d(
                    x=np.full(rows, j),
                    y=np.arange(rows),
                    z=depth[:, j],
                    mode="lines",
                    line=dict(color="#00ff88", width=1),
                    showlegend=False,
                )
            )

        fig.update_layout(
            **self._base_layout("Terrain Wireframe Model"),
            scene=dict(
                xaxis_title="East–West",
                yaxis_title="North–South",
                zaxis_title="Elevation (m)",
                bgcolor="#111111",
            ),
            paper_bgcolor="#111111",
            font=dict(color="white"),
        )
        return fig

    def create_scatter_3d(self, depth: np.ndarray, colorscale: str = "Viridis") -> go.Figure:
        rows, cols = depth.shape
        step = max(1, rows // 30)
        yy, xx = np.mgrid[0:rows:step, 0:cols:step]
        zz = depth[::step, ::step]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xx.ravel(),
                    y=yy.ravel(),
                    z=zz.ravel(),
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=zz.ravel(),
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(title="Elevation (m)"),
                    ),
                )
            ]
        )
        fig.update_layout(
            **self._base_layout("3D Point Cloud — Terrain Intelligence"),
            scene=dict(
                xaxis_title="East–West",
                yaxis_title="North–South",
                zaxis_title="Elevation (m)",
            ),
        )
        return fig

    def terrain_stats(self, depth: np.ndarray) -> dict:
        slope = np.gradient(depth)
        slope_mag = np.sqrt(slope[0] ** 2 + slope[1] ** 2)
        return {
            "min_elevation": round(float(depth.min()), 1),
            "max_elevation": round(float(depth.max()), 1),
            "mean_elevation": round(float(depth.mean()), 1),
            "elevation_range": round(float(depth.max() - depth.min()), 1),
            "mean_slope": round(float(slope_mag.mean()), 2),
            "max_slope": round(float(slope_mag.max()), 2),
            "high_slope_pct": round(float((slope_mag > slope_mag.mean() + slope_mag.std()).mean() * 100), 1),
        }
