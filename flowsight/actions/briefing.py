"""Shift briefing: 5 bullets for the store manager.

If ANTHROPIC_API_KEY is set (and briefing.enabled_if_key), the model rewrites the supplied facts into a
briefing with a strict no-invented-numbers prompt. Otherwise (or on any error) a deterministic template
is used — the briefing always works offline.
"""
from __future__ import annotations

import json
import logging
import os

from flowsight.actions.engine import fmt_dur

log = logging.getLogger(__name__)

SYSTEM = (
    "You are a retail operations assistant writing a shift briefing for a store manager. "
    "Use ONLY the facts in the JSON you are given. Never invent numbers, zones, causes or products. "
    "Write exactly 5 bullet points, each starting with '- ', each under 30 words, most urgent first. "
    "Bullet 1 summarises traffic and conversion. Then the most important actions with their numbers. "
    "End with one opportunity. No preamble, no headings."
)


def _facts(kpis: dict, insights: list[dict], zones: list[dict]) -> dict:
    top_zones = sorted(zones, key=lambda z: -(z.get("visits") or 0))[:5]
    return {
        "kpis": {k: kpis.get(k) for k in ("visitors_total", "people_in_store", "peak_people", "avg_zone_dwell_s",
                                           "avg_time_in_store_s", "checkout_wait_s", "active_bottlenecks",
                                           "bottlenecks_total", "conversion", "elapsed_s")},
        "insights": [{k: i.get(k) for k in ("severity", "title", "action", "expected_impact", "horizon", "owner")}
                     for i in insights[:8]],
        "busiest_zones": [{"name": z.get("name"), "visits": z.get("visits"), "median_dwell_s": z.get("median_dwell_s"),
                           "stopping_power": z.get("stopping_power")} for z in top_zones],
    }


def template_briefing(kpis: dict, insights: list[dict], zones: list[dict]) -> list[str]:
    conv = kpis.get("conversion")
    conv_txt = f"{conv:.0%} entrance→checkout conversion" if conv is not None else "conversion not measurable on this view"
    bullets = [
        f"Traffic: {kpis.get('visitors_total', 0)} shoppers tracked over {fmt_dur(kpis.get('elapsed_s'))}, "
        f"peak {kpis.get('peak_people', 0)} in view, avg zone dwell {kpis.get('avg_zone_dwell_s', 0):.0f}s, {conv_txt}."
    ]
    now_items = [i for i in insights if i.get("horizon") == "now"]
    later = [i for i in insights if i.get("horizon") != "now"]
    for i in (now_items + later)[:3]:
        bullets.append(f"[{i['severity'].upper()} · {i['horizon'].replace('_', ' ')}] {i['action']}")
    while len(bullets) < 4:
        w = kpis.get("checkout_wait_s")
        bullets.append(f"Checkout predicted wait {fmt_dur(w)}; {kpis.get('bottlenecks_total', 0)} congestion "
                       f"episodes so far." if len(bullets) == 1 else "No further urgent actions — floor is flowing normally.")
    hot = [z for z in zones if z.get("attention_status") == "HOT"]
    if hot:
        z = hot[0]
        bullets.append(f"Opportunity: {z['name']} converts {z['stopping_power']:.0%} of passers-by into stops — "
                       f"place promotions next to it.")
    else:
        top = max(zones, key=lambda z: z.get("visits") or 0, default=None)
        bullets.append(f"Opportunity: {top['name']} is the most-visited zone ({top['visits']} visits) — "
                       f"keep it stocked and staffed." if top else "Opportunity: collect more data for zone ranking.")
    return bullets[:5]


def generate_briefing(cfg, kpis: dict, insights: list[dict], zones: list[dict]) -> dict:
    facts = _facts(kpis, insights, zones)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key and cfg.enabled_if_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model=cfg.llm_model, max_tokens=cfg.max_tokens, system=SYSTEM,
                messages=[{"role": "user", "content": "Facts (JSON):\n" + json.dumps(facts, default=str)}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            bullets = [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
            if len(bullets) >= 3:
                return {"mode": "llm", "model": cfg.llm_model, "bullets": bullets[:5], "facts": facts}
            log.warning("LLM briefing malformed; falling back to template")
        except Exception as e:  # network, auth, model name... never break the briefing
            log.warning("LLM briefing failed (%s); using template", e)
    return {"mode": "template", "model": None, "bullets": template_briefing(kpis, insights, zones), "facts": facts}
