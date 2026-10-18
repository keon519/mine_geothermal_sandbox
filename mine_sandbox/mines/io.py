"""Mine files: read and write MineSpec as JSON (or YAML if PyYAML is installed).

A mine file is a plain JSON object:

    {
      "format": "mine-sandbox/mine", "version": 1,
      "name": "...", "description": "...", "source": "...",
      "seam":       {"thickness_m": 2.0, "depth_m": 300, "rock": "coal"},
      "strata":     {"background": "shale",
                     "beds": [{"z_bottom_m": 4, "z_top_m": 1e9, "rock": "sandstone"}]},
      "conditions": {"flow_m3_h": 360, "T_in_C": 35, "T_rock_C": null, ...},
      "pumps":      {"inlet_xy": [0, 3], "outlet_xy": null},
      "grid":       {"plan_cell_m": 6, "layers_above": 5, ...},
      "layout":     {"type": "room_and_pillar", ...}       (see layouts.py)
    }

Rocks are a name from materials.ROCKS or {"name": ..., "k": ..., "rho": ..., "cp": ...}.
Reading is strict: unknown keys and bad values raise MineFileError naming the
exact place in the file, so typos don't silently fall back to defaults.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields

from ..geology import Layer
from ..materials import ROCKS, Rock, get_rock
from .layouts import LAYOUT_TYPES, Layout
from .spec import ConditionsSpec, MineGridSpec, MineSpec, PumpSpec, SeamSpec, StrataSpec

FORMAT, VERSION = "mine-sandbox/mine", 1


class MineFileError(ValueError):
    """A mine file couldn't be read; the message names the file and the exact place in it that's wrong."""
    pass


#  reading
def load_mine(path: str) -> MineSpec:
    """Read a mine file (JSON, or YAML by extension) into a MineSpec, with errors that name the file."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if path.lower().endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError:
                raise MineFileError("reading YAML mine files needs PyYAML (pip install pyyaml)")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except (OSError, json.JSONDecodeError) as e:
        raise MineFileError(f"{path}: {e}") from None
    try:
        return mine_from_dict(data)
    except MineFileError as e:
        raise MineFileError(f"{os.path.basename(path)}: {e}") from None


def mine_from_dict(d: dict) -> MineSpec:
    """Build a MineSpec from the parsed contents of a mine file.

    Only `name` and `layout` are required. Every section is checked for unknown keys and wrong
    types, so a typo fails loudly instead of quietly falling back to a default.
    """
    _check_keys(d, "(top level)", {"format", "version", "name", "description", "source", "seam",
                                   "strata", "conditions", "pumps", "grid", "layout"},
                required={"name", "layout"})
    if d.get("format", FORMAT) != FORMAT:
        raise MineFileError(f"format is '{d.get('format')}', expected '{FORMAT}'")
    if int(d.get("version", VERSION)) > VERSION:
        raise MineFileError(f"file version {d['version']} is newer than this reader ({VERSION})")
    mine = MineSpec(
        name=str(d["name"]),
        description=str(d.get("description", "")),
        source=str(d.get("source", "")),
        layout=read_layout(d["layout"], "layout"),
        seam=_read_seam(d.get("seam", {})),
        strata=_read_strata(d.get("strata", {})),
        conditions=_read_simple(ConditionsSpec, d.get("conditions", {}), "conditions"),
        pumps=_read_pumps(d.get("pumps", {})),
        grid=_read_simple(MineGridSpec, d.get("grid", {}), "grid"),
    )
    if mine.conditions.T_rock_C is None and mine.seam.depth_m is None:
        raise MineFileError("give conditions.T_rock_C, or seam.depth_m to derive it from the geothermal gradient")
    return mine


def read_layout(d: dict, where: str = "layout") -> Layout:
    """Read one layout from file data, looking up its class by the `type` field.

    Composite layouts contain other layouts, so this passes itself down for the nested ones,
    labelling each with its position so errors point at the right part.
    """
    if not isinstance(d, dict) or "type" not in d:
        raise MineFileError(f"{where}: a layout must be an object with a 'type'")
    kind = d["type"]
    cls = LAYOUT_TYPES.get(kind)
    if cls is None:
        raise MineFileError(f"{where}: unknown layout type '{kind}' (known: {sorted(LAYOUT_TYPES)})")
    body = {k: v for k, v in d.items() if k != "type"}
    counter = iter(range(10_000))
    try:
        return cls.from_dict(body, lambda sub: read_layout(sub, f"{where}.{kind}[{next(counter)}]"))
    except MineFileError:
        raise
    except (TypeError, ValueError, KeyError) as e:
        raise MineFileError(f"{where} ({kind}): {e}") from None


def read_rock(v, where: str) -> Rock:
    """A rock given by name, or as a custom {name, k, rho, cp} object."""
    if isinstance(v, dict):
        _check_keys(v, where, {"name", "k", "rho", "cp"}, required={"k", "rho", "cp"})
        return Rock(str(v.get("name", "custom")), float(v["k"]), float(v["rho"]), float(v["cp"]))
    try:
        return get_rock(v)
    except (KeyError, TypeError, ValueError) as e:
        raise MineFileError(f"{where}: {e}") from None


def _read_seam(d):
    """Read the seam section."""
    _check_keys(d, "seam", {"thickness_m", "depth_m", "rock"})
    kw = {k: _num(d[k], f"seam.{k}") for k in ("thickness_m", "depth_m") if d.get(k) is not None}
    if "rock" in d:
        kw["rock"] = read_rock(d["rock"], "seam.rock")
    return _construct(SeamSpec, kw, "seam")


def _read_strata(d):
    """Read the background rock and the beds around the seam, checking each bed's top is above its
    bottom.
    """
    _check_keys(d, "strata", {"background", "beds"})
    beds = []
    for i, b in enumerate(d.get("beds", [])):
        w = f"strata.beds[{i}]"
        _check_keys(b, w, {"z_bottom_m", "z_top_m", "rock"}, required={"z_bottom_m", "z_top_m", "rock"})
        lo, hi = _num(b["z_bottom_m"], w + ".z_bottom_m"), _num(b["z_top_m"], w + ".z_top_m")
        if hi <= lo:
            raise MineFileError(f"{w}: z_top_m must be above z_bottom_m")
        beds.append(Layer(lo, hi, read_rock(b["rock"], w + ".rock")))
    bg = read_rock(d["background"], "strata.background") if "background" in d else ROCKS["shale"]
    return StrataSpec(background=bg, beds=tuple(beds))


def _read_pumps(d):
    """Read the optional pump positions as (x, y) pairs."""
    _check_keys(d, "pumps", {"inlet_xy", "outlet_xy"})
    xy = lambda v, w: None if v is None else tuple(_num(c, w) for c in _pair(v, w))
    return PumpSpec(inlet_xy=xy(d.get("inlet_xy"), "pumps.inlet_xy"),
                    outlet_xy=xy(d.get("outlet_xy"), "pumps.outlet_xy"))


def _read_simple(cls, d, where):
    """Read a section that maps one-to-one onto a dataclass of numbers (conditions, grid)."""
    allowed = {f.name for f in fields(cls)}
    _check_keys(d, where, allowed)
    kw = {}
    for k, v in d.items():
        if v is None:
            kw[k] = None
        elif k == "bridge_diagonals":
            if not isinstance(v, bool):
                raise MineFileError(f"{where}.{k}: expected true or false, got {v!r}")
            kw[k] = v
        elif k in ("margin_cells", "layers_above", "layers_below"):
            kw[k] = int(_num(v, f"{where}.{k}"))
        else:
            kw[k] = _num(v, f"{where}.{k}")
    return _construct(cls, kw, where)


def _construct(cls, kw, where):
    """Create the dataclass, turning its own validation errors into MineFileErrors that say where."""
    try:
        return cls(**kw)
    except (TypeError, ValueError) as e:
        raise MineFileError(f"{where}: {e}") from None


def _check_keys(d, where, allowed, required=frozenset()):
    """Reject non-objects, unknown keys and missing required keys."""
    if not isinstance(d, dict):
        raise MineFileError(f"{where}: expected an object, got {type(d).__name__}")
    unknown = set(d) - set(allowed)
    if unknown:
        raise MineFileError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    missing = set(required) - set(d)
    if missing:
        raise MineFileError(f"{where}: missing required key(s) {sorted(missing)}")


def _num(v, where) -> float:
    """Insist on a real number (booleans don't count, even though Python treats them as ints)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise MineFileError(f"{where}: expected a number, got {v!r}")
    return float(v)


def _pair(v, where):
    """Insist on an [x, y] pair."""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise MineFileError(f"{where}: expected [x, y]")
    return v


#  writing
def _rock_out(r: Rock):
    """Write a standard rock by name, and a custom one with its full properties."""
    return r.name if ROCKS.get(r.name) == r else {"name": r.name, "k": r.k, "rho": r.rho, "cp": r.cp}


def mine_to_dict(m: MineSpec) -> dict:
    """Turn a MineSpec into plain data that `mine_from_dict` will read back to the same mine."""
    simple = lambda obj: {f.name: getattr(obj, f.name) for f in fields(obj)}
    return {
        "format": FORMAT, "version": VERSION,
        "name": m.name, "description": m.description, "source": m.source,
        "seam": {"thickness_m": m.seam.thickness_m, "depth_m": m.seam.depth_m, "rock": _rock_out(m.seam.rock)},
        "strata": {"background": _rock_out(m.strata.background),
                   "beds": [{"z_bottom_m": b.z_bottom, "z_top_m": b.z_top, "rock": _rock_out(b.rock)}
                            for b in m.strata.beds]},
        "conditions": simple(m.conditions),
        "pumps": {"inlet_xy": list(m.pumps.inlet_xy) if m.pumps.inlet_xy else None,
                  "outlet_xy": list(m.pumps.outlet_xy) if m.pumps.outlet_xy else None},
        "grid": simple(m.grid),
        "layout": m.layout.to_dict(),
    }


def save_mine(m: MineSpec, path: str) -> None:
    """Write a mine file, as YAML if the path ends in .yaml/.yml and JSON otherwise."""
    data = mine_to_dict(m)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            import yaml
            yaml.safe_dump(data, f, sort_keys=False)
        else:
            json.dump(data, f, indent=2)
            f.write("\n")
