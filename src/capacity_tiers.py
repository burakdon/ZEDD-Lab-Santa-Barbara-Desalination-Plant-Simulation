#!/usr/bin/env python3

BASE_CAPACITY_AF_MONTH = 3125.0 / 12.0  # Current plant size per month

CAPACITY_TIERS = [
    {"mpd": 3, "vessels": 30, "label": "Tier 1: 3 MPD / 30 vessels"},
    {"mpd": 3, "vessels": 36, "label": "Tier 2: 3 MPD / 36 vessels"},
    {"mpd": 4, "vessels": 30, "label": "Tier 3: 4 MPD / 30 vessels"},
    {"mpd": 4, "vessels": 36, "label": "Tier 4: 4 MPD / 36 vessels"},
    {"mpd": 6, "vessels": 36, "label": "Tier 5: 6 MPD / 36 vessels"},
    {"mpd": 8, "vessels": 36, "label": "Tier 6: 8 MPD / 36 vessels"},
]


def _tier_index_from_p0(p0: float) -> int:
    if p0 >= 1.0:
        return len(CAPACITY_TIERS) - 1
    if p0 <= 0.0:
        return 0
    idx = int(p0 * len(CAPACITY_TIERS))
    if idx == len(CAPACITY_TIERS):
        idx -= 1
    return idx


def _compute_gross_af_month(mpd: float, vessels: float) -> float:
    return BASE_CAPACITY_AF_MONTH * (mpd / 3.0) * (vessels / 30.0)


def get_capacity_tier(p0: float) -> dict:
    idx = _tier_index_from_p0(p0)
    tier = CAPACITY_TIERS[idx]
    gross_month = _compute_gross_af_month(tier["mpd"], tier["vessels"])
    return {
        "label": tier["label"],
        "mpd": tier["mpd"],
        "vessels": tier["vessels"],
        "gross_month": gross_month,
        "gross_annual": gross_month * 12.0,
        "index": idx,
    }

