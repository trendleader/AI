import anthropic
import base64
import json
from datetime import datetime
from typing import Optional


def encode_image(image_data: bytes) -> str:
    return base64.standard_b64encode(image_data).decode("utf-8")


class BaseAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def _vision_request(self, image_data: bytes, prompt: str, system: str = "") -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encode_image(image_data),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if system:
            kwargs["system"] = system
        return self.client.messages.create(**kwargs).content[0].text

    def _text_request(self, prompt: str, system: str = "") -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        return self.client.messages.create(**kwargs).content[0].text


class InterpreterAgent(BaseAgent):
    """Deep technical analysis of agricultural imagery."""

    SYSTEM = """You are an expert agricultural remote sensing analyst with 20 years of experience.
Analyze farm and satellite imagery with precision, identifying vegetation health, irrigation
patterns, crop stress, soil conditions, and anomalies. Use agricultural terminology."""

    def analyze(self, image_data: bytes, geo_context: Optional[dict] = None) -> dict:
        geo_info = ""
        if geo_context:
            geo_info = (
                f"\n\nGeo-context — Lat: {geo_context.get('lat')}, "
                f"Lon: {geo_context.get('lon')}, "
                f"Region: {geo_context.get('region', 'Unknown')}, "
                f"Crop: {geo_context.get('crop_type', 'Unknown')}, "
                f"Soil: {geo_context.get('soil_type', 'Unknown')}, "
                f"Season: {geo_context.get('season', 'Unknown')}, "
                f"Rainfall: {geo_context.get('rainfall_mm', 'N/A')} mm/wk, "
                f"Temp: {geo_context.get('temp_c', 'N/A')}°C"
            )

        prompt = f"""Analyze this agricultural/satellite image and return a structured report:

1. VEGETATION_HEALTH: Rate 0-10 with detailed description
2. IRRIGATION_STATUS: Distribution efficiency and coverage
3. CROP_PATTERNS: Type, density, uniformity observations
4. STRESS_INDICATORS: Water, nutrient, disease, pest signs
5. SOIL_CONDITION: Visible soil health indicators
6. ANOMALIES: List each with format — Description | Severity: LOW/MEDIUM/HIGH/CRITICAL
7. NDVI_ESTIMATE: Estimated NDVI range (0.0–1.0)
8. RECOMMENDATIONS: Top 3 prioritized immediate actions{geo_info}"""

        raw = self._vision_request(image_data, prompt, self.SYSTEM)
        return {
            "raw_analysis": raw,
            "anomalies": self._parse_anomalies(raw),
            "timestamp": datetime.now().isoformat(),
        }

    def _parse_anomalies(self, text: str) -> list:
        anomalies = []
        in_section = False
        for line in text.split("\n"):
            if "ANOMALIES" in line.upper():
                in_section = True
                continue
            if in_section:
                if any(
                    s in line.upper() for s in ["SOIL_", "NDVI_", "RECOMMENDATION"]
                ):
                    break
                for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                    if sev in line.upper() and line.strip():
                        anomalies.append(
                            {"description": line.strip(), "severity": sev}
                        )
                        break
        return anomalies


class ExplainerAgent(BaseAgent):
    """Translates technical analysis into plain-language insights."""

    SYSTEM = """You are a friendly, empathetic agricultural advisor. Convert technical
analysis into clear, actionable guidance tailored to your audience. Be solution-focused
and encouraging while highlighting genuine risks."""

    AUDIENCES = {
        "farmer": "a working farmer with practical knowledge but limited technical background",
        "agronomist": "a professional agronomist with strong technical expertise",
        "investor": "an agricultural investor focused on yield, ROI, and risk",
        "researcher": "an agricultural researcher interested in detailed scientific findings",
    }

    def explain(self, technical_analysis: str, audience: str = "farmer") -> str:
        context = self.AUDIENCES.get(audience, self.AUDIENCES["farmer"])
        prompt = f"""Given this technical agricultural analysis:

{technical_analysis}

Rewrite for {context}. Include:
1. **Summary** (2–3 sentences) of key findings
2. **Practical Impact** — what this means for their operation
3. **Immediate Actions** — top 3, prioritized
4. **Watch List** — what to monitor over the next 7 days
5. **Positive Note** — what is working well"""
        return self._text_request(prompt, self.SYSTEM)


class AnomalyDetectorAgent(BaseAgent):
    """Specialized anomaly detection and classification agent."""

    SYSTEM = """You are a precision agriculture anomaly detection AI. Identify and classify
visual anomalies in agricultural imagery with clinical precision. Severity scale:
CRITICAL (act within 24h), HIGH (act within 3 days), MEDIUM (act within 2 weeks),
LOW (monitor only). Always include OVERALL_HEALTH_SCORE and ALERT_REQUIRED fields."""

    def detect(self, image_data: bytes) -> dict:
        prompt = """Perform anomaly detection on this agricultural image.

For EACH anomaly found, provide:
- ANOMALY_TYPE: disease | drought | pest | nutrient | mechanical | erosion | other
- LOCATION: describe position in frame
- SEVERITY: CRITICAL/HIGH/MEDIUM/LOW
- URGENCY: DAYS/WEEKS/MONTHS
- AFFECTED_AREA: estimated % of visible area
- RECOMMENDED_ACTION: specific intervention

Then provide:
OVERALL_HEALTH_SCORE: [0-100]
ALERT_REQUIRED: YES/NO
PRIORITY_ZONES: describe areas needing immediate attention"""

        result = self._vision_request(image_data, prompt, self.SYSTEM)
        health_score = self._extract_score(result)
        alert = "ALERT_REQUIRED: YES" in result.upper()

        return {
            "full_report": result,
            "health_score": health_score,
            "alert_required": alert,
            "timestamp": datetime.now().isoformat(),
        }

    def _extract_score(self, text: str) -> int:
        for line in text.split("\n"):
            if "OVERALL_HEALTH_SCORE" in line.upper():
                try:
                    digits = "".join(filter(str.isdigit, line.split(":")[-1].strip()[:3]))
                    if digits:
                        return min(100, max(0, int(digits)))
                except Exception:
                    pass
        return 70


class WhatIfAgent(BaseAgent):
    """Answers hypothetical 'what-if' agricultural questions with visual context."""

    SYSTEM = """You are an expert agricultural decision-support AI. Answer hypothetical
'what-if' questions using evidence from the visible image. Provide probability estimates,
timelines, and risk-adjusted recommendations. Be specific and actionable."""

    def answer(
        self,
        image_data: bytes,
        question: str,
        context: Optional[dict] = None,
    ) -> str:
        ctx_str = f"\n\nAdditional context: {json.dumps(context, indent=2)}" if context else ""
        prompt = f"""Viewing this agricultural image, answer the following what-if question:

QUESTION: {question}{ctx_str}

Provide:
1. **SCENARIO ANALYSIS** — what would happen based on visible conditions
2. **PROBABILITY** — likelihood this outcome occurs (0–100%)
3. **TIMELINE** — when effects would manifest
4. **BEST CASE** — optimal outcome
5. **WORST CASE** — maximum risk scenario
6. **MITIGATION** — how to optimize for the best outcome
7. **CONFIDENCE** — LOW/MEDIUM/HIGH confidence in this prediction"""
        return self._vision_request(image_data, prompt, self.SYSTEM)

    def stream(self, image_data: bytes, question: str):
        """Yield streamed response tokens for real-time display."""
        prompt = (
            f"Viewing this agricultural image, answer: {question}\n\n"
            "Provide scenario analysis, probability, timeline, best/worst case, and mitigation."
        )
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=self.SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encode_image(image_data),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        ) as stream:
            yield from stream.text_stream
