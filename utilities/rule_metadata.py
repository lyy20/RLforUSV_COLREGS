"""Stable metadata encodings used by the optional rule replay buffer.

The values in this module are part of the checkpoint/data contract.  Do not use
``hash()`` for these fields because Python deliberately randomizes hash seeds
between processes.
"""

from __future__ import annotations


ENCOUNTER_TYPE_CODES = {
    "unknown": 0,
    "safe_passage": 1,
    "head_on": 2,
    "crossing_give_way": 3,
    "crossing_stand_on": 4,
    "overtaking_give_way": 5,
    "overtaken_stand_on": 6,
    "collision_risk_undefined": 7,
}
ENCOUNTER_TYPE_NAMES = {value: key for key, value in ENCOUNTER_TYPE_CODES.items()}


# Keep a fixed order for all modes emitted by the current filter.  The final
# ``unknown`` entry is used for future/legacy values that are not recognized.
FILTER_MODE_NAMES = (
    "none",
    "disabled",
    "pass_through",
    "network",
    "colregs_head_on",
    "colregs_crossing_give_way",
    "colregs_overtaking_give_way",
    "colregs_give_way",
    "colregs_stand_on_emergency",
    "colregs_stand_on",
    "colregs_undefined_risk",
    "unknown",
)
FILTER_MODE_CODES = {name: index for index, name in enumerate(FILTER_MODE_NAMES)}
FILTER_MODE_NAMES_BY_CODE = {value: key for key, value in FILTER_MODE_CODES.items()}


def _normalized_name(value, fallback="unknown"):
    if value is None:
        return fallback
    if hasattr(value, "value"):
        value = value.value
    name = str(value).strip().lower()
    return name if name else fallback


def encode_encounter_type(value) -> int:
    """Encode a COLREGs encounter name to a stable integer."""
    name = _normalized_name(value)
    # Some legacy reports use the enum value ``crossing``.  It is safer to
    # preserve the fact that a rule event existed than to silently drop it.
    aliases = {
        "crossing": "crossing_give_way",
        "overtaking": "overtaking_give_way",
        "undefined_risk": "collision_risk_undefined",
        "collision_risk": "collision_risk_undefined",
    }
    return int(ENCOUNTER_TYPE_CODES.get(aliases.get(name, name), 0))


def decode_encounter_type(code) -> str:
    try:
        return ENCOUNTER_TYPE_NAMES.get(int(round(float(code))), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def encode_filter_mode(value) -> int:
    """Encode a filter mode without relying on process-dependent hashing."""
    name = _normalized_name(value, fallback="none")
    return int(FILTER_MODE_CODES.get(name, FILTER_MODE_CODES["unknown"]))


def decode_filter_mode(code) -> str:
    try:
        return FILTER_MODE_NAMES_BY_CODE.get(int(round(float(code))), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def rule_type_mapping():
    """Return a serializable copy for checkpoints and run manifests."""
    return {
        "encounter_type_codes": dict(ENCOUNTER_TYPE_CODES),
        "filter_mode_codes": dict(FILTER_MODE_CODES),
    }
