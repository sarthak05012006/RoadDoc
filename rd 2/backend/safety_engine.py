"""
ROADDOC extension — Vehicle Monitoring, Tyre Monitoring and the
Safety Decision Engine described in the TechBirds pitch deck.

This module is intentionally dependency-free (no torch/numpy needed) so it
can run anywhere the Flask app runs. It takes the existing road-condition
result (label / wetness_score / trend, produced by track_analyzer.py or
model_service.py) and combines it with vehicle + tyre telemetry to produce
the four warning types + one general recommendation called out in the
pitch deck's "Safety Decision Engine" slide.
"""
from __future__ import annotations
from typing import Optional

VEHICLE_TYPES = ["car", "suv", "bike", "bus", "truck"]
DRIVING_CONDITIONS = ["normal", "aggressive", "highway", "city", "off-road"]
TYRE_CONDITIONS = ["new", "good", "worn", "bald", "damaged"]

# Base "safe" top speed (km/h) per road condition, before vehicle-type and
# driving-condition adjustments. These are illustrative, not certified
# values — good enough for a hackathon prototype / demo.
BASE_SAFE_SPEED = {
    "dry": 120,
    "damp": 90,
    "drying": 100,
    "wet": 70,
    "standing water": 40,
}

VEHICLE_SPEED_FACTOR = {
    "car": 1.0,
    "suv": 0.95,
    "bike": 0.8,
    "bus": 0.75,
    "truck": 0.7,
}

DRIVING_CONDITION_FACTOR = {
    "normal": 1.0,
    "aggressive": 0.85,   # aggressive driving needs a bigger safety margin
    "highway": 1.05,
    "city": 0.9,
    "off-road": 0.8,
}


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def standing_water_alert(wetness_score: float) -> bool:
    """Very high wetness + saturated reflection reads like standing water.

    The trained classifier / heuristic only ever emit Dry/Damp/Drying/Wet,
    so this is applied as a derived, application-layer signal on top of the
    wetness score rather than a fifth training class.
    """
    return wetness_score is not None and wetness_score >= 0.88


def road_condition_warning(label: str, trend: str, water_alert: bool) -> str:
    label_l = (label or "").lower()
    if water_alert:
        return "⚠️ Standing water suspected on the surface ahead — high aquaplaning risk."
    if label_l == "wet":
        return "Wet surface detected. Grip is significantly reduced."
    if label_l == "drying":
        return "Surface is transitioning (drying). Expect uneven grip patch-to-patch."
    if label_l == "damp":
        return "Damp surface detected. Grip is moderately reduced."
    if label_l == "dry" and trend == "Getting wetter":
        return "Surface currently dry, but conditions are worsening — stay alert."
    return "Surface condition is dry and stable."


class VehicleMonitor:
    """Packages vehicle telemetry and derives a condition-aware safe-speed
    envelope (pitch deck component 2)."""

    def __init__(self, vehicle_type="car", speed=None, location=None, driving_condition="normal"):
        self.vehicle_type = (vehicle_type or "car").lower()
        if self.vehicle_type not in VEHICLE_SPEED_FACTOR:
            self.vehicle_type = "car"
        self.speed = _safe_float(speed)
        self.location = location or None
        self.driving_condition = (driving_condition or "normal").lower()
        if self.driving_condition not in DRIVING_CONDITION_FACTOR:
            self.driving_condition = "normal"

    def recommended_max_speed(self, road_label: str, water_alert: bool) -> int:
        key = "standing water" if water_alert else (road_label or "dry").lower()
        base = BASE_SAFE_SPEED.get(key, BASE_SAFE_SPEED["dry"])
        factor = VEHICLE_SPEED_FACTOR[self.vehicle_type] * DRIVING_CONDITION_FACTOR[self.driving_condition]
        return round(base * factor)

    def speed_warning(self, road_label: str, water_alert: bool) -> Optional[str]:
        max_speed = self.recommended_max_speed(road_label, water_alert)
        if self.speed is None:
            return f"No live speed reported. Recommended max speed for current conditions: {max_speed} km/h."
        if self.speed > max_speed:
            over_by = round(self.speed - max_speed)
            return (f"Speed too high for conditions: travelling at {round(self.speed)} km/h, "
                    f"recommended max is {max_speed} km/h ({over_by} km/h over).")
        return f"Speed within safe range ({round(self.speed)} km/h ≤ {max_speed} km/h recommended max)."

    def as_dict(self):
        return {
            "vehicle_type": self.vehicle_type,
            "speed_kmh": self.speed,
            "location": self.location,
            "driving_condition": self.driving_condition,
        }


class TyreMonitor:
    """Packages tyre telemetry and flags pressure / condition issues
    (pitch deck component 3)."""

    def __init__(self, pressure=None, recommended_pressure=None, condition=None, temperature=None):
        self.pressure = _safe_float(pressure)
        self.recommended_pressure = _safe_float(recommended_pressure)
        self.condition = (condition or None)
        if self.condition:
            self.condition = self.condition.lower()
            if self.condition not in TYRE_CONDITIONS:
                self.condition = None
        self.temperature = _safe_float(temperature)

    def pressure_status(self):
        """Returns one of: 'unknown', 'low', 'high', 'critical-low', 'ok'."""
        if self.pressure is None or self.recommended_pressure is None or self.recommended_pressure == 0:
            return "unknown"
        deviation = (self.pressure - self.recommended_pressure) / self.recommended_pressure
        if deviation <= -0.20:
            return "critical-low"
        if deviation <= -0.10:
            return "low"
        if deviation >= 0.15:
            return "high"
        return "ok"

    def pressure_warning(self):
        status = self.pressure_status()
        if status == "unknown":
            return "Tyre pressure not reported — recommend checking manually before the next run."
        if status == "critical-low":
            return (f"⚠️ Critically low tyre pressure ({self.pressure} PSI vs recommended "
                     f"{self.recommended_pressure} PSI). Aquaplaning and blowout risk increases sharply — "
                     "reduce speed and check tyres immediately.")
        if status == "low":
            return (f"Tyre pressure is low ({self.pressure} PSI vs recommended "
                     f"{self.recommended_pressure} PSI). Grip and handling margin reduced.")
        if status == "high":
            return (f"Tyre pressure is high ({self.pressure} PSI vs recommended "
                     f"{self.recommended_pressure} PSI). Contact patch is reduced, especially on wet surfaces.")
        return f"Tyre pressure is within range ({self.pressure} PSI vs recommended {self.recommended_pressure} PSI)."

    def inspection_recommendation(self, road_label: str, water_alert: bool):
        reasons = []
        status = self.pressure_status()
        if status in ("low", "critical-low", "high"):
            reasons.append("pressure out of range")
        if self.condition in ("worn", "bald", "damaged"):
            reasons.append(f"tyre condition reported as '{self.condition}'")
        if self.temperature is not None and self.temperature >= 100:
            reasons.append(f"tyre temperature high ({self.temperature}°C)")
        if water_alert or (road_label or "").lower() == "wet":
            if self.condition in ("worn", "bald", "damaged") or status in ("low", "critical-low"):
                reasons.append("wet/standing-water surface amplifies existing tyre risk")
        if not reasons:
            if self.condition in (None,):
                return "Tyre condition not reported — a visual check is recommended before extended running."
            return "No tyre inspection needed right now — condition and pressure look fine."
        return "Tyre inspection recommended: " + "; ".join(reasons) + "."

    def as_dict(self):
        return {
            "pressure_psi": self.pressure,
            "recommended_pressure_psi": self.recommended_pressure,
            "condition": self.condition,
            "temperature_c": self.temperature,
            "pressure_status": self.pressure_status(),
        }


RISK_ORDER = ["low", "medium", "high", "critical"]


def _risk_level(water_alert, road_label, tyre: TyreMonitor, speed_over):
    score = 0
    label_l = (road_label or "").lower()
    if water_alert:
        score += 3
    elif label_l == "wet":
        score += 2
    elif label_l == "drying":
        score += 1
    elif label_l == "damp":
        score += 1
    tyre_status = tyre.pressure_status()
    if tyre_status == "critical-low":
        score += 3
    elif tyre_status in ("low", "high"):
        score += 1
    if tyre.condition in ("bald", "damaged"):
        score += 2
    elif tyre.condition == "worn":
        score += 1
    if speed_over:
        score += 2
    if score >= 6:
        return "critical"
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def general_recommendation(risk_level: str) -> str:
    return {
        "low": "Conditions are stable — continue normal driving with routine monitoring.",
        "medium": "Stay attentive: road and/or tyre conditions are starting to work against you.",
        "high": "Reduce speed and increase following distance now — multiple risk factors are stacking up.",
        "critical": "Slow down immediately and consider pulling over to check tyres — conditions and vehicle state are both compromised.",
    }.get(risk_level, "Monitoring conditions.")


def decide(road_result: dict, vehicle: VehicleMonitor, tyre: TyreMonitor) -> dict:
    """The Safety Decision Engine (pitch deck component 4): combines road,
    vehicle and tyre signals into the warning set + recommendation."""
    label = road_result.get("label", "Dry")
    trend = road_result.get("trend", "Not enough data yet")
    wetness_score = road_result.get("wetness_score", 0.0)
    water_alert = standing_water_alert(wetness_score)

    speed_warn = vehicle.speed_warning(label, water_alert)
    max_speed = vehicle.recommended_max_speed(label, water_alert)
    speed_over = vehicle.speed is not None and vehicle.speed > max_speed

    risk = _risk_level(water_alert, label, tyre, speed_over)

    return {
        "standing_water_alert": water_alert,
        "risk_level": risk,
        "road_condition_warning": road_condition_warning(label, trend, water_alert),
        "tyre_pressure_warning": tyre.pressure_warning(),
        "speed_related_warning": speed_warn,
        "recommended_max_speed_kmh": max_speed,
        "tyre_inspection_recommendation": tyre.inspection_recommendation(label, water_alert),
        "general_safety_recommendation": general_recommendation(risk),
        "vehicle": vehicle.as_dict(),
        "tyre": tyre.as_dict(),
    }
