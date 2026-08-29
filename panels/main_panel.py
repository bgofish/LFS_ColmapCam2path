"""Main panel for the ColmapCamPath plugin.

Converts a COLMAP sparse reconstruction (cameras.txt/.bin + images.txt/.bin,
anywhere under a chosen folder) into an LFS camera-path JSON, then loads it
straight into the Sequencer -- automatically backing up whatever path was
there first.

Coordinate conversion mirrors the standalone converter this plugin replaces:

  COLMAP  ->  LFS (SuperSplat)
  +X right    +X right
  +Y down     +Y up      <- flipped
  +Z forward  +Z forward (SuperSplat +Z = camera forward)

Steps per image:
  1. Invert COLMAP R_cw  ->  R_wc  (world-to-camera -> camera-to-world)
  2. Convert position:  t_world = -R_cw^T . t_colmap
  3. Flip Y axis: pos_lfs = (x, -y, z)
  4. Build R_lfs from R_wc with Y-flip: negate rows/cols touching Y
  5. Roll about the camera's own forward (camera->target) axis (default
     +90 deg -- LFS's camera convention needs this relative to COLMAP's)
  6. Convert R_lfs to quaternion (SuperSplat +Z-forward convention)
"""

import json
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import lichtfeld as lf

_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# ── Paths ────────────────────────────────────────────────────────────────────
_PLUGIN_DIR    = Path(__file__).resolve().parent.parent
_DEFAULTS_PATH = _PLUGIN_DIR / "DEFAULTS.JSON"
_SCRIPTS_DIR = (
    Path(os.environ.get("USERPROFILE", "~")).expanduser()
    / ".lichtfeld" / "plugins" / "ColmapCamPath" / "Scripts"
)
_BACKUP_DIR = _SCRIPTS_DIR / "backups"

MODEL_NAME = "colmap_campath"

# (name, is_int, min, max, step)
_FIELD_SPEC_ORDER = [
    ("fps",        False,  0.1,   30.0, 0.1),
    ("scale",      False, 0.1, 10, 0.1),
    ("sensor_mm",  False, 1.0,   100.0, 0.5),
    ("roll_deg",   False, -180.0, 180.0, 1.0),
]
_FIELD_VALUE_DEFAULTS = {
    "fps": 1.0, "scale": 1.0000, "sensor_mm": 36.0, "roll_deg": 63.0,
}

_HARDCODED_DEFAULTS = {
    "output_filename": "colmap_camera_path.json",
    "fields": {
        name: {"value": _FIELD_VALUE_DEFAULTS[name], "min": lo, "max": hi, "step": step, "int": is_int}
        for name, is_int, lo, hi, step in _FIELD_SPEC_ORDER
    },
}


def _load_defaults() -> dict:
    """Load DEFAULTS.JSON from the plugin root if present, filling in any
    missing/invalid keys from the built-in fallback above -- a partial,
    malformed, or absent file never breaks the panel."""
    merged = json.loads(json.dumps(_HARDCODED_DEFAULTS))  # deep copy

    try:
        with open(_DEFAULTS_PATH, "r", encoding="utf-8") as f:
            user = json.load(f)
    except FileNotFoundError:
        return merged
    except Exception as e:
        lf.log.error(f"ColmapCamPath: DEFAULTS.JSON is invalid, using built-in defaults ({e})")
        return merged

    if "output_filename" in user:
        merged["output_filename"] = user["output_filename"]

    user_fields = user.get("fields", {})
    if isinstance(user_fields, dict):
        for name, spec in user_fields.items():
            if name not in merged["fields"] or not isinstance(spec, dict):
                continue
            for k in ("value", "min", "max", "step", "int"):
                if k in spec:
                    merged["fields"][name][k] = spec[k]

    return merged


# ── COLMAP binary/text readers ────────────────────────────────────────────────

def _read_cameras_bin(path: Path) -> dict:
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id, model_id, w, h = struct.unpack("<IiQQ", f.read(24))
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 8, 7: 5}.get(model_id, 4)
            params = list(struct.unpack(f"<{nparams}d", f.read(8 * nparams)))
            cameras[cam_id] = {"model_id": model_id, "width": w, "height": h, "params": params}
    return cameras


def _read_cameras_txt(path: Path) -> dict:
    cameras = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            w, h = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {"width": w, "height": h, "params": params}
    return cameras


def _read_images_bin(path: Path) -> list:
    images = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz, tx, ty, tz = struct.unpack("<7d", f.read(56))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)
            images.append({"qw": qw, "qx": qx, "qy": qy, "qz": qz,
                           "tx": tx, "ty": ty, "tz": tz, "cam_id": cam_id, "name": name})
    return images


def _read_images_txt(path: Path) -> list:
    images = []
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        images.append({"qw": qw, "qx": qx, "qy": qy, "qz": qz,
                        "tx": tx, "ty": ty, "tz": tz, "cam_id": cam_id, "name": name})
        i += 2  # skip the 2D-point line
    return images


# ── Quaternion / matrix helpers ───────────────────────────────────────────────

def _quat_to_matrix(qw, qx, qy, qz):
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def _mat_transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _rot_z(deg):
    """Rotation matrix about the LOCAL +Z axis (camera-to-target / forward axis)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _matrix_to_quat(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2][1] - m[1][2]) * s
        qy = (m[0][2] - m[2][0]) * s
        qz = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        qw = (m[2][1] - m[1][2]) / s; qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s; qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        qw = (m[0][2] - m[2][0]) / s; qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s; qz = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        qw = (m[1][0] - m[0][1]) / s; qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s; qz = 0.25 * s
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / n, qx / n, qy / n, qz / n


def _focal_to_mm(focal_px, sensor_px, sensor_mm):
    return focal_px * sensor_mm / sensor_px


def _natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


# ── Core conversion ───────────────────────────────────────────────────────────

def convert(images_path: Path, cameras_path: Path, fps: float, sensor_mm: float,
            focal_override_mm, scale: float, roll_deg: float ) -> dict:
    """Convert a COLMAP sparse model to an LFS camera-path dict."""
    cameras = _read_cameras_bin(cameras_path) if cameras_path.suffix == ".bin" else _read_cameras_txt(cameras_path)
    images = _read_images_bin(images_path) if images_path.suffix == ".bin" else _read_images_txt(images_path)

    if not images:
        raise ValueError("No images found in the images file.")

    images.sort(key=lambda im: _natural_key(im["name"]))

    keyframes = []
    for idx, im in enumerate(images):
        time_sec = round(idx / fps, 6)

        cam = cameras.get(im["cam_id"], cameras.get(next(iter(cameras))))
        params = cam["params"]
        focal_px = params[0]
        w = cam["width"]
        fl_mm = focal_override_mm if focal_override_mm is not None else round(_focal_to_mm(focal_px, w, sensor_mm), 4)

        qw, qx, qy, qz = im["qw"], im["qx"], im["qy"], im["qz"]
        tx, ty, tz = im["tx"], im["ty"], im["tz"]

        R_cw = _quat_to_matrix(qw, qx, qy, qz)
        R_wc = _mat_transpose(R_cw)
        t_colmap = [tx, ty, tz]
        t_world = [-sum(R_wc[i][j] * t_colmap[j] for j in range(3)) for i in range(3)]

        px = t_world[0] * scale
        py = -t_world[1] * scale
        pz = -t_world[2] * scale

        def _to_lfs_rot(m):
            r = [[0.0] * 3 for _ in range(3)]
            for row in range(3):
                sy = -1 if row == 1 else 1
                r[row][0] = sy * m[row][0]
                r[row][1] = sy * m[row][1]
                r[row][2] = -sy * m[row][2]
            return r

        R_lfs = _to_lfs_rot(R_wc)
        if roll_deg:
            R_lfs = _mat_mul(R_lfs, _rot_z(roll_deg))
        rqw, rqx, rqy, rqz = _matrix_to_quat(R_lfs)

        keyframes.append({
            "easing": 0,
            "focal_length_mm": fl_mm,
            "position": [round(px, 6), round(py, 6), round(pz, 6)],
            "rotation": [round(rqw, 6), round(rqx, 6), round(rqy, 6), round(rqz, 6)],
            "time": time_sec,
        })

    return {"keyframes": keyframes, "version": 3}


# ── File discovery ────────────────────────────────────────────────────────────

def _find_in_tree(d: Path, stem: str, max_depth: int = 5):
    """Search a directory tree (bounded depth) for '<stem>.bin' first, then
    '<stem>.txt', preferring the shallowest match. Handles any COLMAP
    layout -- project/sparse/0/, project/distorted/sparse/0/, or the files
    sitting right at the root."""
    for ext in (".bin", ".txt"):
        target = f"{stem}{ext}"
        best = None
        for root, dirs, files in os.walk(d):
            depth = len(Path(root).relative_to(d).parts)
            if depth > max_depth:
                dirs[:] = []
                continue
            if target in files:
                candidate = Path(root) / target
                if best is None or depth < best[0]:
                    best = (depth, candidate)
        if best is not None:
            return best[1]
    return None


def find_colmap_files(folder: str):
    """Given a folder path (project root, sparse/0, or anywhere in between),
    find the images and cameras files. Returns (images_path, cameras_path)
    or raises FileNotFoundError."""
    d = Path(folder).expanduser()
    if not d.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")

    images_path = _find_in_tree(d, "images")
    cameras_path = _find_in_tree(d, "cameras")

    if images_path is None:
        raise FileNotFoundError("Could not find images.txt or images.bin under that folder")
    if cameras_path is None:
        raise FileNotFoundError("Could not find cameras.txt or cameras.bin under that folder")

    return images_path, cameras_path


# ── Native folder dialog ──────────────────────────────────────────────────────
# LFS's plugin API doesn't expose its own file-dialog call, and tkinter isn't
# reliably usable inside the embedded Python runtime (no working display
# context in-process). The proven approach other LFS plugins use instead:
# shell out to a short PowerShell script that pops a native Windows Forms
# dialog and prints the chosen path to stdout. Windows-only; other platforms
# fall back to typing/pasting the path into the Folder field directly.

def _select_folder_dialog(title: str, initial_dir: str = ""):
    if sys.platform != "win32":
        return None
    safe_initial = initial_dir.replace('"', '""')
    ps_script = f'''
    Add-Type -AssemblyName System.Windows.Forms
    $d = New-Object System.Windows.Forms.FolderBrowserDialog
    $d.Description = "{title}"
    $d.ShowNewFolderButton = $false
    if ("{safe_initial}" -ne "") {{ $d.SelectedPath = "{safe_initial}" }}
    if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
        Write-Output $d.SelectedPath
    }}
    '''
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True,
            creationflags=_SUBPROCESS_FLAGS,
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception as e:
        lf.log.warning(f"ColmapCamPath: folder dialog failed — {e}")
        return None


# ── Panel ─────────────────────────────────────────────────────────────────────

class MainPanel(lf.ui.Panel):
    """COLMAP -> LFS camera path converter."""

    id          = "ColmapCamPath.main_panel"
    label       = "COLMAP to LFS"
    space       = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order       = 110
    template    = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT

    def __init__(self):
        d = _load_defaults()
        f = d["fields"]

        self._colmap_folder = ""
        self._images_path = None   # resolved by last successful Convert/Browse
        self._cameras_path = None
        self._found_text = ""

        self._numeric_specs = {}
        for name, is_int, lo, hi, step in _FIELD_SPEC_ORDER:
            spec = f.get(name, {})
            fis_int = bool(spec.get("int", is_int))
            fmin = float(spec.get("min", lo))
            fmax = float(spec.get("max", hi))
            fstep = float(spec.get("step", step))
            fval = float(spec.get("value", _FIELD_VALUE_DEFAULTS[name]))
            if fis_int:
                fval = float(int(round(fval)))

            float_attr = f"_{name}"
            text_attr = f"_{name}_text"
            setattr(self, float_attr, fval)
            setattr(self, text_attr, self._fmt_num(fval, fis_int))

            self._numeric_specs[name] = dict(
                float_attr=float_attr, text_attr=text_attr,
                is_int=fis_int, min=fmin, max=fmax, step=fstep,
            )

        self._focal_override_text = ""  # blank = derive from COLMAP intrinsics
        self._output_path = str(_SCRIPTS_DIR / d.get("output_filename", "colmap_camera_path.json"))

        self._status = ""
        self._status_ok = True
        self._last_backup_text = ""

        self._sec_source    = True
        self._sec_options   = True
        self._sec_output    = True
        self._sec_sequencer = True

        self._handle = None

    # ------------------------------------------------------------------
    # Retained data model
    # ------------------------------------------------------------------

    _SIMPLE_FIELDS = [
        ("colmap_folder", "_colmap_folder", str),
        ("focal_override_text", "_focal_override_text", str),
        ("output_path", "_output_path", str),
        ("sec_source", "_sec_source", bool),
        ("sec_options", "_sec_options", bool),
        ("sec_output", "_sec_output", bool),
        ("sec_sequencer", "_sec_sequencer", bool),
    ]

    def _bind_two_way(self, model, name, attr, cast):
        def setter(v, a=attr, c=cast):
            try:
                setattr(self, a, c(v))
            except (TypeError, ValueError):
                pass
        model.bind(name, lambda a=attr: getattr(self, a), setter)

    @staticmethod
    def _fmt_num(v: float, is_int: bool) -> str:
        if is_int:
            return str(int(round(v)))
        v = round(v, 4)
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-") else "0"

    def _bind_numeric(self, model, name, float_attr, text_attr, is_int):
        def slider_get(fa=float_attr):
            return getattr(self, fa)

        def slider_set(v, fa=float_attr, ta=text_attr, ii=is_int, nm=name):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return
            if ii:
                fv = float(int(round(fv)))
            setattr(self, fa, fv)
            setattr(self, ta, self._fmt_num(fv, ii))
            if self._handle is not None:
                self._handle.dirty(f"{nm}_text")

        model.bind(name, slider_get, slider_set)

        def text_get(ta=text_attr):
            return getattr(self, ta)

        def text_set(v, fa=float_attr, ta=text_attr, ii=is_int, nm=name):
            setattr(self, ta, str(v))
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return
            if ii:
                fv = float(int(round(fv)))
            setattr(self, fa, fv)
            if self._handle is not None:
                self._handle.dirty(nm)

        model.bind(f"{name}_text", text_get, text_set)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model(MODEL_NAME)
        if model is None:
            return

        for name, attr, cast in self._SIMPLE_FIELDS:
            self._bind_two_way(model, name, attr, cast)

        for name, spec in self._numeric_specs.items():
            self._bind_numeric(model, name, spec["float_attr"], spec["text_attr"], spec["is_int"])

        model.bind_func("status", lambda: self._status)
        model.bind_func("status_ok", lambda: self._status_ok)
        model.bind_func("found_text", lambda: self._found_text)
        model.bind_func("last_backup_text", lambda: self._last_backup_text)

        self._handle = model.get_handle()

    def _set_status(self, msg: str, ok: bool = True):
        self._status = msg
        self._status_ok = ok
        if self._handle is not None:
            self._handle.dirty_all()

    # ------------------------------------------------------------------
    # DOM wiring
    # ------------------------------------------------------------------

    def on_mount(self, doc):
        browse_btn = doc.get_element_by_id("btn-browse")
        if browse_btn:
            browse_btn.add_event_listener("click", lambda _ev: self._do_browse())

        convert_btn = doc.get_element_by_id("btn-convert")
        if convert_btn:
            convert_btn.add_event_listener("click", lambda _ev: self._do_convert())

        load_btn = doc.get_element_by_id("btn-load")
        if load_btn:
            load_btn.add_event_listener("click", lambda _ev: self._do_load())

        backup_btn = doc.get_element_by_id("btn-backup")
        if backup_btn:
            backup_btn.add_event_listener("click", lambda _ev: self._do_backup())

        try:
            field_inputs = doc.query_selector_all(".field-input")
        except Exception:
            field_inputs = []
        for el in field_inputs:
            def _select_all(_ev, el=el):
                try:
                    el.select()
                except Exception:
                    pass
            el.add_event_listener("focus", _select_all)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _do_browse(self):
        """Open a native folder-picker for the COLMAP source folder.

        Uses a PowerShell + Windows Forms dialog rather than tkinter --
        tkinter isn't reliably usable inside LFS's embedded Python runtime
        (no working display context in-process), which is why the earlier
        tkinter-based version of this button didn't work. Windows-only; on
        other platforms this tells the user to paste the path in instead.
        """
        initial = self._colmap_folder if self._colmap_folder and Path(self._colmap_folder).is_dir() else str(Path.home())
        selected = _select_folder_dialog("Select COLMAP folder", initial)

        if sys.platform != "win32":
            self._set_status(
                "Native folder browser is only available on Windows here — paste the COLMAP folder path into the field above instead.",
                False,
            )
            return

        if not selected:
            return  # user cancelled, or the dialog failed silently

        self._colmap_folder = selected
        if self._handle is not None:
            self._handle.dirty("colmap_folder")
        self._try_locate()

    def _try_locate(self):
        """Attempt to resolve cameras/images files under the current folder
        and report what was found, without doing a full conversion."""
        folder = self._colmap_folder.strip()
        if not folder:
            return
        try:
            images_path, cameras_path = find_colmap_files(folder)
        except Exception as e:
            self._images_path = None
            self._cameras_path = None
            self._found_text = ""
            self._set_status(f"⚠ {e}", False)
            return

        self._images_path = images_path
        self._cameras_path = cameras_path
        self._found_text = f"Found {images_path.name} + {cameras_path.name} in {images_path.parent}"
        self._set_status(f"✓ Located COLMAP files — ready to convert.")

    def _do_convert(self):
        folder = self._colmap_folder.strip()
        if not folder:
            self._set_status("⚠ Set a COLMAP folder first (Browse… or paste a path).", False)
            return

        try:
            images_path, cameras_path = find_colmap_files(folder)
        except Exception as e:
            self._set_status(f"⚠ {e}", False)
            return

        out = self._output_path.strip()
        if not out:
            self._set_status("⚠ No output path set.", False)
            return

        focal_str = self._focal_override_text.strip()
        try:
            focal_override = float(focal_str) if focal_str else None
        except ValueError:
            self._set_status(f"⚠ Focal override must be a number (got {focal_str!r}).", False)
            return

        try:
            data = convert(
                images_path, cameras_path,
                fps=self._fps, sensor_mm=self._sensor_mm,
                focal_override_mm=focal_override, scale=self._scale,
                roll_deg=self._roll_deg,
            )
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as e:
            self._set_status(f"⚠ Conversion failed: {e}", False)
            lf.log.error(f"ColmapCamPath: conversion failed — {e}")
            return

        self._images_path = images_path
        self._cameras_path = cameras_path
        self._found_text = f"Found {images_path.name} + {cameras_path.name} in {images_path.parent}"

        kf_count = len(data["keyframes"])
        self._set_status(f"✓ Converted {kf_count} keyframes → {out}")
        lf.log.info(f"ColmapCamPath: wrote {kf_count} keyframes to {out!r}")

    def _do_load(self):
        """Backs up whatever is currently in the Sequencer, then loads the
        converted JSON in — so a bad conversion never destroys existing work."""
        out = self._output_path.strip()
        if not out or not Path(out).exists():
            self._set_status("⚠ Nothing to load yet — run Convert first.", False)
            return

        backup_path = self._write_backup()

        try:
            lf.ui.load_camera_path(out)
        except Exception as e:
            self._set_status(f"⚠ Load failed: {e}", False)
            return

        if backup_path:
            self._set_status(f"✓ Backed up previous path → {backup_path.name}, then loaded {out}")
        else:
            self._set_status(f"✓ Loaded into Sequencer: {out}  (no existing path to back up)")

    def _do_backup(self):
        backup_path = self._write_backup()
        if backup_path:
            self._set_status(f"✓ Sequencer backed up → {backup_path}")
        else:
            self._set_status("Backup skipped — is there a camera path in the sequencer?", False)

    def _write_backup(self):
        """Save the current Sequencer path to a timestamped file under
        Scripts/backups/. Returns the Path written, or None if there was
        nothing to back up (or the host reported failure)."""
        import datetime
        try:
            _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            backup_path = _BACKUP_DIR / f"backup_{stamp}.json"
            ok = lf.ui.save_camera_path(str(backup_path))
            if not ok:
                return None
            self._last_backup_text = f"Last backup: {backup_path}"
            if self._handle is not None:
                self._handle.dirty("last_backup_text")
            return backup_path
        except Exception as e:
            lf.log.warning(f"ColmapCamPath: backup failed — {e}")
            return None
