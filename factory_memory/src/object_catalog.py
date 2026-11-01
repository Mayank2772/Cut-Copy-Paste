"""
Factory Memory object catalogue for workshop / fabrication videos.

- YOLO_PORTABLE_CLASSES: prompts sent to YOLO-World (canonical names only).
- Aliases map noisy detector strings back to one stable name.
- Shape / size metadata is soft context only — never used to drop a detection.
"""

OBJECT_CATALOG = {
    # ----- power tools (distinct silhouettes) -----
    "angle grinder": {
        "aliases": ["grinder", "disk grinder", "side grinder"],
        "category": "power_tool", "shape": "compact_power_tool",
        "aspect_ratio": (0.6, 2.2), "relative_size": "small_or_medium",
    },
    "orbital sander": {
        "aliases": ["power sander", "sander", "palm sander", "random orbit sander"],
        "category": "power_tool", "shape": "compact_power_tool",
        "aspect_ratio": (0.6, 2.0), "relative_size": "small_or_medium",
    },
    "power drill": {
        "aliases": ["cordless drill", "drill", "electric drill", "impact driver"],
        "category": "power_tool", "shape": "elongated_power_tool",
        "aspect_ratio": (1.2, 4.0), "relative_size": "small_or_medium",
    },
    "circular saw": {
        "aliases": ["power saw", "skill saw"],
        "category": "power_tool", "shape": "power_tool_with_blade",
        "aspect_ratio": (0.8, 2.5), "relative_size": "medium",
    },
    "jigsaw": {
        "aliases": ["sabre saw", "reciprocating saw"],
        "category": "power_tool", "shape": "elongated_power_tool",
        "aspect_ratio": (1.2, 3.5), "relative_size": "small_or_medium",
    },
    "nail gun": {
        "aliases": ["nailer", "brad nailer", "staple gun"],
        "category": "power_tool", "shape": "compact_power_tool",
        "aspect_ratio": (0.8, 2.5), "relative_size": "small_or_medium",
    },
    "router": {
        "aliases": ["wood router"],
        "category": "power_tool", "shape": "compact_power_tool",
        "aspect_ratio": (0.6, 2.0), "relative_size": "small_or_medium",
    },

    # ----- hand tools -----
    "hammer": {
        "aliases": ["claw hammer", "mallet"],
        "category": "hand_tool", "shape": "tool_with_head",
        "aspect_ratio": (1.0, 5.0), "relative_size": "small",
    },
    "screwdriver": {
        "aliases": ["screw driver", "phillips screwdriver"],
        "category": "hand_tool", "shape": "elongated_tool",
        "aspect_ratio": (2.5, 12.0), "relative_size": "small",
    },
    "wrench": {
        "aliases": ["spanner", "adjustable wrench"],
        "category": "hand_tool", "shape": "elongated_tool",
        "aspect_ratio": (2.0, 10.0), "relative_size": "small",
    },
    "pliers": {
        "aliases": ["needle nose pliers", "cutting pliers"],
        "category": "hand_tool", "shape": "hand_tool",
        "aspect_ratio": (1.5, 6.0), "relative_size": "small",
    },
    "clamp": {
        "aliases": ["C-clamp", "bar clamp", "spring clamp", "G-clamp"],
        "category": "hand_tool", "shape": "clamp",
        "aspect_ratio": (0.5, 3.0), "relative_size": "small_or_medium",
    },
    "tape measure": {
        "aliases": ["measuring tape", "tape-measure"],
        "category": "hand_tool", "shape": "compact_tool",
        "aspect_ratio": (0.5, 3.0), "relative_size": "small",
    },
    "hand saw": {
        "aliases": ["wood saw", "panel saw"],
        "category": "hand_tool", "shape": "elongated_tool",
        "aspect_ratio": (2.0, 12.0), "relative_size": "medium",
    },

    # ----- marking / consumables -----
    "marker pen": {
        "aliases": ["marker", "felt pen", "permanent marker", "pen"],
        "category": "small_tool", "shape": "thin_elongated",
        "aspect_ratio": (3.0, 15.0), "relative_size": "small",
    },
    "pencil": {
        "aliases": ["carpenter pencil", "wood pencil"],
        "category": "small_tool", "shape": "thin_elongated",
        "aspect_ratio": (4.0, 20.0), "relative_size": "small",
    },
    "glue bottle": {
        "aliases": ["glue", "adhesive bottle", "wood glue"],
        "category": "container", "shape": "bottle",
        "aspect_ratio": (0.35, 2.2), "relative_size": "small",
    },
    "sandpaper": {
        "aliases": ["sand paper", "abrasive paper"],
        "category": "consumable", "shape": "flat_sheet",
        "aspect_ratio": (0.5, 3.0), "relative_size": "small",
    },
    "sanding block": {
        "aliases": ["sand block"],
        "category": "hand_tool", "shape": "block",
        "aspect_ratio": (0.5, 2.5), "relative_size": "small",
    },

    # ----- wood / material -----
    "wooden plank": {
        "aliases": ["timber", "wooden board", "wood plank", "lumber"],
        "category": "wood_material", "shape": "long_rectangular",
        "aspect_ratio": (1.5, 15.0), "relative_size": "medium_or_large",
    },
    "wooden piece": {
        "aliases": ["wood block", "wood piece", "plywood piece", "wooden block"],
        "category": "wood_material", "shape": "compact_rectangular",
        "aspect_ratio": (0.4, 3.0), "relative_size": "small_or_medium",
    },
    "plywood sheet": {
        "aliases": ["plywood board", "plywood panel"],
        "category": "wood_material", "shape": "flat_sheet",
        "aspect_ratio": (0.8, 8.0), "relative_size": "medium_or_large",
    },

    # ----- stone (kept for other clips) -----
    "granite slab": {
        "aliases": ["granite piece", "stone slab", "granite panel"],
        "category": "stone_material", "shape": "flat_slab",
        "aspect_ratio": (1.2, 10.0), "relative_size": "medium_or_large",
    },

    # ----- packaging / PPE / misc -----
    "glove": {
        "aliases": ["work glove", "safety glove", "gloves"],
        "category": "ppe", "shape": "soft_irregular",
        "aspect_ratio": (0.5, 2.5), "relative_size": "small",
    },
    "box": {
        "aliases": ["cardboard box", "carton", "packing box"],
        "category": "container", "shape": "box_like",
        "aspect_ratio": (0.5, 2.5), "relative_size": "medium",
    },
    "power cord": {
        "aliases": ["electric cable", "electrical cable", "extension cord", "cable"],
        "category": "electrical", "shape": "thin_flexible_line",
        "aspect_ratio": (2.0, 30.0), "relative_size": "small_or_long",
    },
}

# Priority boost when confidences are close (distinctive tools win over generic wood/box)
CLASS_PRIORITY = {
    "angle grinder": 4, "orbital sander": 4, "circular saw": 4,
    "power drill": 3, "nail gun": 3, "jigsaw": 3, "router": 3,
    "hammer": 2, "screwdriver": 2, "marker pen": 2, "glue bottle": 2,
    "clamp": 2, "tape measure": 1, "wooden plank": 2, "plywood sheet": 1,
    "wooden piece": 0, "box": 0, "glove": 1, "power cord": 0,
}

YOLO_PORTABLE_CLASSES = list(OBJECT_CATALOG.keys())

FIXED = [
    "workbench", "table", "wooden frame", "shelf", "machine",
    "floor", "wall", "cabinet", "bench", "storage rack",
]


def normalize_name(name):
    return " ".join(str(name).lower().replace("_", " ").strip().split())


_ALIAS_TO_CANONICAL = {}
for _canonical, _profile in OBJECT_CATALOG.items():
    _ALIAS_TO_CANONICAL[normalize_name(_canonical)] = _canonical
    for _alias in _profile.get("aliases", []):
        _ALIAS_TO_CANONICAL[normalize_name(_alias)] = _canonical


def canonicalize_name(name):
    normalized = normalize_name(name)
    return _ALIAS_TO_CANONICAL.get(normalized, str(name).strip())


def get_profile(name):
    canonical = canonicalize_name(name)
    return canonical, OBJECT_CATALOG.get(canonical)


def enrich_detection(obj, frame_width, frame_height):
    """Attach measured bbox geometry and soft catalogue metadata (never drops dets)."""
    x1, y1, x2, y2 = obj["bbox"]
    width = max(1.0, float(x2) - float(x1))
    height = max(1.0, float(y2) - float(y1))
    frame_area = max(1.0, float(frame_width) * float(frame_height))
    canonical, profile = get_profile(obj["name"])
    ratio = width / height
    expected = profile["aspect_ratio"] if profile else None
    out = dict(obj)
    out["name"] = canonical
    out["bbox_width"] = round(width, 1)
    out["bbox_height"] = round(height, 1)
    out["aspect_ratio"] = round(ratio, 3)
    out["relative_frame_area"] = round(width * height / frame_area, 6)
    out["expected_shape"] = profile["shape"] if profile else "unknown"
    out["expected_relative_size"] = profile["relative_size"] if profile else "unknown"
    out["shape_match"] = (
        expected[0] <= ratio <= expected[1] if expected else None
    )
    out["class_priority"] = CLASS_PRIORITY.get(canonical, 0)
    return out
