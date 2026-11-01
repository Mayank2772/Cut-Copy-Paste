"""
Factory Memory object reference catalogue for workshop / fabrication videos.

Shape and relative-size profiles are weak visual priors only. They must never
suppress a YOLO detection or decide whether a hand is grasping an object.
"""

OBJECT_CATALOG = {
    # Stone / granite workpieces
    "granite slab": {
        "aliases": ["granite piece", "stone slab", "granite panel", "stone panel"],
        "category": "stone_material", "shape": "flat_slab",
        "aspect_ratio": (1.2, 10.0), "relative_size": "medium_or_large",
    },
    "granite strip": {
        "aliases": ["stone strip", "granite sample strip"],
        "category": "stone_material", "shape": "long_flat_strip",
        "aspect_ratio": (2.0, 15.0), "relative_size": "small_or_medium",
    },
    "marble slab": {
        "aliases": ["marble piece", "marble panel"],
        "category": "stone_material", "shape": "flat_slab",
        "aspect_ratio": (1.2, 10.0), "relative_size": "medium_or_large",
    },

    # Extrusions, profiles, pipes and channels
    "aluminium profile": {
        "aliases": ["aluminum profile", "aluminium extrusion", "metal extrusion",
                    "aluminium channel", "metal channel", "aluminium rail"],
        "category": "metal_profile", "shape": "long_rigid_profile",
        "aspect_ratio": (3.0, 25.0), "relative_size": "long",
    },
    "metal pipe": {
        "aliases": ["metal tube", "steel pipe", "steel tube"],
        "category": "pipe_tube", "shape": "long_cylindrical_or_hollow",
        "aspect_ratio": (3.0, 25.0), "relative_size": "long",
    },
    "plastic pipe": {
        "aliases": ["PVC pipe", "PVC tube", "plastic tube"],
        "category": "pipe_tube", "shape": "long_cylindrical_or_hollow",
        "aspect_ratio": (3.0, 25.0), "relative_size": "long",
    },
    "plastic profile": {
        "aliases": ["PVC profile", "plastic extrusion", "plastic channel"],
        "category": "plastic_profile", "shape": "long_rigid_profile",
        "aspect_ratio": (3.0, 25.0), "relative_size": "long",
    },
    "metal bracket": {
        "aliases": ["metal fitting", "metal connector", "mounting bracket"],
        "category": "metal_hardware", "shape": "small_angular_part",
        "aspect_ratio": (0.4, 5.0), "relative_size": "small",
    },

    # Lighting components
    "LED strip": {
        "aliases": ["LED light strip", "LED tape light", "flexible LED strip"],
        "category": "lighting_component", "shape": "thin_flexible_strip",
        "aspect_ratio": (4.0, 30.0), "relative_size": "small_or_long",
    },
    "LED tube light": {
        "aliases": ["LED batten", "linear LED light", "LED light bar"],
        "category": "lighting_component", "shape": "long_light_bar",
        "aspect_ratio": (3.0, 18.0), "relative_size": "medium_or_long",
    },
    "light diffuser cover": {
        "aliases": ["LED diffuser", "light cover", "diffuser strip",
                    "translucent light cover", "light lens"],
        "category": "lighting_component", "shape": "long_translucent_cover",
        "aspect_ratio": (2.0, 20.0), "relative_size": "medium_or_long",
    },
    "light fixture housing": {
        "aliases": ["light housing", "LED housing", "light box",
                    "LED profile housing"],
        "category": "lighting_component", "shape": "rigid_light_enclosure",
        "aspect_ratio": (1.0, 15.0), "relative_size": "medium_or_large",
    },

    # Rolls, tapes and flexible coverings
    "adhesive tape roll": {
        "aliases": ["tape roll", "packing tape", "clear tape", "sticky tape"],
        "category": "tape_and_roll", "shape": "circular_roll",
        "aspect_ratio": (0.65, 1.55), "relative_size": "small",
    },
    "masking tape roll": {
        "aliases": ["paper tape roll", "masking tape"],
        "category": "tape_and_roll", "shape": "circular_roll",
        "aspect_ratio": (0.65, 1.55), "relative_size": "small",
    },
    "insulation tape roll": {
        "aliases": ["electrical tape", "electrician tape"],
        "category": "tape_and_roll", "shape": "small_circular_roll",
        "aspect_ratio": (0.65, 1.55), "relative_size": "small",
    },
    "protective plastic film": {
        "aliases": ["plastic covering", "protective film", "wrapping film",
                    "transparent plastic sheet"],
        "category": "covering_material", "shape": "thin_flexible_sheet",
        "aspect_ratio": (0.5, 12.0), "relative_size": "variable",
    },
    "foam sheet": {
        "aliases": ["foam covering", "foam padding", "protective foam"],
        "category": "covering_material", "shape": "flat_soft_sheet",
        "aspect_ratio": (0.5, 12.0), "relative_size": "medium_or_large",
    },
    "fabric covering": {
        "aliases": ["cloth covering", "protective cloth", "fabric sheet"],
        "category": "covering_material", "shape": "flexible_sheet",
        "aspect_ratio": (0.5, 12.0), "relative_size": "variable",
    },

    # Cables, small components and packaging
    "electrical cable": {
        "aliases": ["wire", "electric wire", "power cable", "cable bundle"],
        "category": "electrical_component", "shape": "thin_flexible_line",
        "aspect_ratio": (2.0, 30.0), "relative_size": "small_or_long",
    },
    "connector": {
        "aliases": ["electrical connector", "wire connector", "plug connector"],
        "category": "electrical_component", "shape": "small_connector",
        "aspect_ratio": (0.4, 3.0), "relative_size": "small",
    },
    "screw": {
        "aliases": ["bolt", "fastener"],
        "category": "fastener", "shape": "tiny_elongated_metal_part",
        "aspect_ratio": (1.5, 10.0), "relative_size": "tiny",
    },
    "small parts tray": {
        "aliases": ["parts container", "component tray"],
        "category": "container", "shape": "shallow_open_container",
        "aspect_ratio": (0.6, 3.0), "relative_size": "small_or_medium",
    },
    "plastic container": {
        "aliases": ["plastic tub", "plastic cup", "small plastic box"],
        "category": "container", "shape": "container",
        "aspect_ratio": (0.4, 2.5), "relative_size": "small_or_medium",
    },
    "cardboard box": {
        "aliases": ["carton", "packing box"],
        "category": "container", "shape": "box_like",
        "aspect_ratio": (0.5, 2.5), "relative_size": "medium_or_large",
    },

    # Existing general workshop objects
    "wooden plank": {
        "aliases": ["timber", "wooden board", "wood plank"],
        "category": "wood_material", "shape": "long_rectangular",
        "aspect_ratio": (1.5, 15.0), "relative_size": "medium_or_large",
    },
    "wood block": {
        "aliases": ["wooden piece", "wood piece"],
        "category": "wood_material", "shape": "compact_rectangular",
        "aspect_ratio": (0.5, 2.5), "relative_size": "small_or_medium",
    },
    "hammer": {
        "aliases": [], "category": "hand_tool", "shape": "tool_with_head",
        "aspect_ratio": (1.0, 5.0), "relative_size": "small",
    },
    "screwdriver": {
        "aliases": ["screw driver"], "category": "hand_tool",
        "shape": "elongated_tool", "aspect_ratio": (2.5, 10.0),
        "relative_size": "small",
    },
    "marker pen": {
        "aliases": ["marker", "pen"], "category": "small_tool",
        "shape": "thin_elongated", "aspect_ratio": (3.0, 15.0),
        "relative_size": "small",
    },
    "glue bottle": {
        "aliases": ["glue container"], "category": "container",
        "shape": "bottle", "aspect_ratio": (0.4, 2.0),
        "relative_size": "small_or_medium",
    },
}

# Use one canonical prompt per class. Aliases are resolved after detection,
# avoiding duplicate boxes caused by listing synonyms as separate YOLO classes.
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
    for _alias in _profile["aliases"]:
        _ALIAS_TO_CANONICAL[normalize_name(_alias)] = _canonical


def canonicalize_name(name):
    normalized = normalize_name(name)
    return _ALIAS_TO_CANONICAL.get(normalized, str(name).strip())


def get_profile(name):
    canonical = canonicalize_name(name)
    return canonical, OBJECT_CATALOG.get(canonical)


def enrich_detection(obj, frame_width, frame_height):
    """Attach measured bbox geometry and soft catalogue metadata."""
    x1, y1, x2, y2 = obj["bbox"]
    width = max(1.0, float(x2) - float(x1))
    height = max(1.0, float(y2) - float(y1))
    frame_area = max(1.0, float(frame_width) * float(frame_height))
    canonical, profile = get_profile(obj["name"])
    ratio = width / height
    expected = profile.get("aspect_ratio") if profile else None
    return {
        **obj,
        "raw_name": obj["name"],
        "name": canonical,
        "canonical_name": canonical,
        "category": profile["category"] if profile else "unclassified",
        "bbox_width_px": round(width, 1),
        "bbox_height_px": round(height, 1),
        "aspect_ratio": round(ratio, 3),
        "relative_frame_area": round(width * height / frame_area, 6),
        "expected_shape": profile["shape"] if profile else "unknown",
        "expected_relative_size": profile["relative_size"] if profile else "unknown",
        "shape_match": (
            expected[0] <= ratio <= expected[1] if expected else None
        ),
    }
