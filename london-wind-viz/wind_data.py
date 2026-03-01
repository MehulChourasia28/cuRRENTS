"""
Fetch real-time wind data from OpenWeather across a spatial grid and
extrapolate to multiple heights via the logarithmic wind profile.
"""
import asyncio
import math
import logging
from dataclasses import dataclass, field

import config

log = logging.getLogger(__name__)

SURFACE_ROUGHNESS_Z0 = 1.5   # urban roughness length (m)
REFERENCE_HEIGHT_M = 10       # OpenWeather reports at 10 m
PROFILE_HEIGHTS = [1.5, 5.0, 10.0, 20.0, 50.0]


@dataclass
class WindProfile:
    """Domain-averaged wind conditions at multiple heights."""
    speed_at_height: dict[float, float] = field(default_factory=dict)
    direction_deg: float = 0.0
    center_lat: float = 0.0
    center_lon: float = 0.0
    num_samples: int = 0


def log_wind_profile(u_ref: float, z: float,
                     z_ref: float = REFERENCE_HEIGHT_M,
                     z0: float = SURFACE_ROUGHNESS_Z0) -> float:
    """Log-law: u(z) = u_ref * ln(z/z0) / ln(z_ref/z0)."""
    if z <= z0 or z_ref <= z0:
        return 0.0
    return u_ref * math.log(z / z0) / math.log(z_ref / z0)


async def fetch_wind_profile(center_lat: float, center_lon: float,
                             half_x_m: float, half_y_m: float,
                             grid_n: int = 5) -> WindProfile:
    """Sample OpenWeather at a grid_n x grid_n grid across the domain
    and return an averaged multi-height wind profile."""
    import aiohttp

    api_key = config.OPENWEATHER_API_KEY
    if not api_key:
        log.warning("No OPENWEATHER_API_KEY — using fallback wind profile")
        return _fallback_profile(center_lat, center_lon)

    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(center_lat)))

    points = []
    for iy in range(grid_n):
        for ix in range(grid_n):
            dx = -half_x_m + (ix + 0.5) * (2 * half_x_m / grid_n)
            dy = -half_y_m + (iy + 0.5) * (2 * half_y_m / grid_n)
            points.append((
                center_lat + dy * lat_per_m,
                center_lon + dx * lon_per_m,
            ))

    speeds_10m = []
    directions = []

    async with aiohttp.ClientSession() as session:
        for lat, lon in points:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={lat:.6f}&lon={lon:.6f}"
                f"&appid={api_key}&units=metric"
            )
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    speeds_10m.append(data["wind"]["speed"])
                    directions.append(data["wind"].get("deg", 0))
            except Exception as exc:
                log.debug("OpenWeather point (%f,%f) failed: %s", lat, lon, exc)
            await asyncio.sleep(0.15)

    if not speeds_10m:
        log.warning("All OpenWeather requests failed — using fallback")
        return _fallback_profile(center_lat, center_lon)

    avg_speed_10m = sum(speeds_10m) / len(speeds_10m)
    avg_dir = _circular_mean(directions)

    profile = WindProfile(
        direction_deg=round(avg_dir, 1),
        center_lat=center_lat,
        center_lon=center_lon,
        num_samples=len(speeds_10m),
    )
    for z in PROFILE_HEIGHTS:
        profile.speed_at_height[z] = round(
            log_wind_profile(avg_speed_10m, z), 2)

    log.info("Wind profile from %d samples: 10m=%.1f m/s @ %.0f°",
             len(speeds_10m), avg_speed_10m, avg_dir)
    for z, s in sorted(profile.speed_at_height.items()):
        log.info("  %5.1f m → %.2f m/s", z, s)

    return profile


def _circular_mean(angles_deg: list[float]) -> float:
    """Average of angles on a circle."""
    if not angles_deg:
        return 0.0
    sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360


def _fallback_profile(lat: float, lon: float) -> WindProfile:
    avg_10m = 4.5
    profile = WindProfile(
        direction_deg=225.0,
        center_lat=lat,
        center_lon=lon,
        num_samples=0,
    )
    for z in PROFILE_HEIGHTS:
        profile.speed_at_height[z] = round(
            log_wind_profile(avg_10m, z), 2)
    return profile
