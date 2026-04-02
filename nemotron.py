"""
Nemotron-based wind scenario generation and reward-model validation.

Adapted from NVIDIAhackathon/pipeline.py.  Generates batches of plausible
urban wind variations via nvidia/llama-3.1-nemotron-nano-8b-v1 and scores
each with nvidia/llama-3.1-nemotron-70b-reward.  Keeps generating until
the target number of validated scenarios is reached.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

import config
from wind_data import WindProfile

log = logging.getLogger(__name__)

GENERATION_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"
REWARD_MODEL = "nvidia/llama-3.1-nemotron-70b-reward"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

BATCH_SIZE = 5
MAX_ROUNDS = 6
MAX_GEN_RETRIES = 4
REWARD_CONCURRENCY = 3
RETRY_BASE_DELAY = 2.0

REWARD_MIDPOINT = -24.0
REWARD_SCALE = 2.0


@dataclass
class WindScenario:
    u_ref: float            # speed at 50 m (m/s)
    direction_deg: float    # absolute wind direction (degrees)
    direction_offset: float
    correctness: float
    justification: str


class _RateLimitError(Exception):
    pass


class _NIMClient:
    def __init__(self) -> None:
        self._c = AsyncOpenAI(base_url=NIM_BASE_URL,
                              api_key=config.NIM_API_KEY)

    @retry(retry=retry_if_exception_type(_RateLimitError),
           wait=wait_exponential(multiplier=2, min=4, max=120),
           stop=stop_after_attempt(8))
    async def chat(self, model: str, messages: list[dict], *,
                   temperature: float = 0.7, max_tokens: int = 4096) -> str:
        try:
            r = await self._c.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            return r.choices[0].message.content or ""
        except Exception as exc:
            m = str(exc).lower()
            if "429" in m or "rate" in m or "too many" in m:
                raise _RateLimitError(str(exc)) from exc
            raise

    async def close(self) -> None:
        await self._c.close()


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _repair_json(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    text = re.sub(r"[\x00-\x1f]+", " ", text)
    ob = text.count("{") - text.count("}")
    oq = text.count("[") - text.count("]")
    if ob > 0 or oq > 0:
        lc = text.rfind("}")
        if lc != -1:
            text = text[:lc + 1]
            oq = text.count("[") - text.count("]")
            text += "]" * oq
    return text


def _parse_scenarios(raw: str) -> list[dict]:
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"```[a-zA-Z]*\s*\n?", "", cleaned).strip()
    m = _JSON_ARRAY_RE.search(cleaned)
    candidate = m.group() if m else cleaned
    for inp in [candidate, _repair_json(candidate)]:
        try:
            arr = json.loads(inp)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            continue
    obj_re = re.compile(
        r'"u_ref"\s*:\s*([\d.]+).*?'
        r'"direction_offset"\s*:\s*([-\d.]+).*?'
        r'"justification"\s*:\s*"([^"]*)"', re.DOTALL)
    matches = obj_re.findall(re.sub(r"//[^\n]*", "", candidate))
    return [{"u_ref": float(u), "direction_offset": float(d),
             "justification": j} for u, d, j in matches]


_GEN_PROMPT = """\
You are an urban wind-engineering expert creating synthetic CFD boundary \
conditions for a dense urban area at ({lat:.4f} N, {lon:.4f} W).

**Current macro-weather (scaled to 50 m via log-law, z0=1.5 m):**
- Reference wind speed at 50 m: {speed_50m} m/s
- Wind direction: {direction_deg}°

**Task:** Generate exactly {batch_size} physically plausible wind-scenario \
variations reflecting urban micro-meteorological phenomena (channelling, \
Venturi acceleration, vortex shedding, wake effects, corner effects, \
thermal updrafts, pressure differentials).

For each variation provide:
1. `u_ref` – wind speed at 50 m (m/s), within ±60 % of {speed_50m}.
2. `direction_offset` – deviation from macro direction (−45 to +45 deg).
3. `justification` – one sentence explaining the physical mechanism.

Return ONLY a raw JSON array.  No markdown fences, no code comments.
[{{"u_ref": 7.2, "direction_offset": -12.3, "justification": "Venturi …"}}, …]"""


async def _generate_batch(client: _NIMClient,
                          profile: WindProfile) -> list[dict]:
    speed_50m = profile.speed_at_height.get(50.0, 6.0)
    prompt = _GEN_PROMPT.format(
        lat=profile.center_lat, lon=abs(profile.center_lon),
        speed_50m=speed_50m, direction_deg=profile.direction_deg,
        batch_size=BATCH_SIZE)

    for attempt in range(MAX_GEN_RETRIES):
        try:
            raw = await client.chat(GENERATION_MODEL,
                                    [{"role": "user", "content": prompt}],
                                    temperature=0.9, max_tokens=8192)
            items = _parse_scenarios(raw)
            valid = []
            for s in items:
                try:
                    valid.append({
                        "u_ref": float(s["u_ref"]),
                        "direction_offset": float(s["direction_offset"]),
                        "justification": str(s["justification"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            if valid:
                log.info("  → parsed %d scenarios (attempt %d)",
                         len(valid), attempt + 1)
                return valid
        except _RateLimitError:
            raise
        except Exception as exc:
            log.warning("  → gen error (attempt %d): %s", attempt + 1, exc)
        await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    return []


def _parse_reward(raw: str) -> Optional[float]:
    raw = raw.strip().lower()
    m = re.search(r"reward\s*:\s*([-+]?\d*\.?\d+)", raw)
    if m:
        return float(m.group(1))
    try:
        return float(raw)
    except ValueError:
        return None


def _normalise(raw_reward: float) -> float:
    x = (raw_reward - REWARD_MIDPOINT) / REWARD_SCALE
    return 1.0 / (1.0 + math.exp(-x))


async def _score_one(client: _NIMClient, profile: WindProfile,
                     scenario: dict, sem: asyncio.Semaphore,
                     idx: int) -> Optional[WindScenario]:
    speed_50m = profile.speed_at_height.get(50.0, 6.0)
    user_msg = (f"Urban wind at 50 m (z0=1.5 m): u_ref={speed_50m} m/s, "
                f"direction={profile.direction_deg}°. "
                f"Provide a plausible urban wind variation.")
    asst_msg = (f"u_ref={scenario['u_ref']} m/s, "
                f"direction_offset={scenario['direction_offset']}°. "
                f"Justification: {scenario['justification']}")
    messages = [{"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg}]

    async with sem:
        for attempt in range(5):
            try:
                raw = await client.chat(REWARD_MODEL, messages,
                                        temperature=0.0, max_tokens=64)
                score = _parse_reward(raw)
                if score is None:
                    return None
                correctness = _normalise(score)
                abs_dir = (profile.direction_deg
                           + scenario["direction_offset"]) % 360
                return WindScenario(
                    u_ref=scenario["u_ref"],
                    direction_deg=round(abs_dir, 1),
                    direction_offset=scenario["direction_offset"],
                    correctness=round(correctness, 4),
                    justification=scenario["justification"],
                )
            except _RateLimitError:
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            except Exception as exc:
                log.warning("Reward #%d error: %s", idx, exc)
                await asyncio.sleep(RETRY_BASE_DELAY)
    return None


async def _score_batch(client: _NIMClient, profile: WindProfile,
                       scenarios: list[dict]) -> list[WindScenario]:
    sem = asyncio.Semaphore(REWARD_CONCURRENCY)
    tasks = [_score_one(client, profile, s, sem, i)
             for i, s in enumerate(scenarios, 1)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def generate_variations(profile: WindProfile) -> list[WindScenario]:
    """Generate and validate wind variations until NEMOTRON_TARGET pass."""
    target = config.NEMOTRON_TARGET
    threshold = config.NEMOTRON_THRESHOLD
    client = _NIMClient()
    validated: list[WindScenario] = []

    for rnd in range(1, MAX_ROUNDS + 1):
        remaining = target - len(validated)
        if remaining <= 0:
            break
        log.info("Nemotron round %d | validated %d / %d",
                 rnd, len(validated), target)

        batch = await _generate_batch(client, profile)
        if not batch:
            continue

        scored = await _score_batch(client, profile, batch)
        passing = [s for s in scored if s.correctness > threshold]
        log.info("  %d generated → %d scored → %d passed (>%.1f)",
                 len(batch), len(scored), len(passing), threshold)

        space = target - len(validated)
        validated.extend(passing[:space])

    await client.close()
    log.info("Nemotron: %d validated scenarios collected", len(validated))
    return validated
