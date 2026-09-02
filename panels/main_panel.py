"""Main panel for the ColmapCamPath plugin.

Reads cameras already loaded in the LFS scene (camera_R, camera_T,
camera_focal_x, camera_width), converts to LFS Sequencer keyframe format,
and optionally skips cameras disabled for training.

Coordinate conversion (verified against LFS probe):
  Position : C = -R_cw^T · t_cw  →  LFS_pos = (Cx, -Cy, -Cz)
  Rotation : LFS_rot = (qw, -qx, qy, qz)  where (qw,qx,qy,qz) from R_cw
"""

import json
import math
import re
import sys
from pathlib import Path

import lichtfeld as lf

# ── Paths ─────────────────────────────────────────────────────────────────────
_PLUGIN_DIR  = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path.home() / ".lichtfeld" / "plugins" / "ColmapCamPath" / "Scripts"
_BACKUP_DIR  = _SCRIPTS_DIR / "backups"

MODEL_NAME = "colmap_campath"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def _focal_to_mm(focal_px, sensor_px, sensor_mm=36.0):
    return focal_px * sensor_mm / sensor_px if sensor_px > 0 else 50.0


def _quat_to_matrix(qw, qx, qy, qz):
    return [
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ]


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
        qy = 0.25 * s;                 qz = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        qw = (m[1][0] - m[0][1]) / s; qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s; qz = 0.25 * s
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    return qw/n, qx/n, qy/n, qz/n


def _tensor_to_list(t):
    """Convert a lichtfeld.Tensor to a plain Python list (1-D or 2-D)."""
    if hasattr(t, "tolist"):
        return t.tolist()
    return list(t)


# ── Training-enabled state ────────────────────────────────────────────────────

def _training_enabled_set() -> set:
    """Return set of camera names enabled for training, or None on failure."""
    try:
        scene = lf.get_scene()
        return {
            node.name
            for node in scene.get_nodes()
            if str(node.type) == "NodeType.CAMERA" and node.training_enabled
        }
    except Exception as e:
        lf.log.warn(f"ColmapCamPath: could not read training states — {e}")
        return None


def _get_scene_cameras(skip_disabled: bool) -> list:
    """Return list of camera dicts in natural-sort order.

    Each dict: name, position (LFS space), rotation (qw,qx,qy,qz LFS),
    focal_mm.  Cameras disabled for training are excluded when
    skip_disabled=True.
    """
    try:
        scene = lf.get_scene()
        cams  = scene.get_active_cameras() if scene is not None else []
    except Exception as e:
        lf.log.warn(f"ColmapCamPath: get_active_cameras() failed — {e}")
        cams = []

    if not cams:
        raise ValueError("No cameras found in the scene — load a COLMAP dataset into LFS first.")

    enabled = _training_enabled_set() if skip_disabled else None

    result = []
    skipped = 0
    for cam in cams:
        name = getattr(cam, "name", "?")

        # Training filter
        if enabled is not None and name not in enabled:
            skipped += 1
            continue

        try:
            R_raw = _tensor_to_list(cam.camera_R)
            t_raw = _tensor_to_list(cam.camera_T)

            # Normalise R to list-of-3-rows regardless of flat vs nested
            if R_raw and not isinstance(R_raw[0], (list, tuple)):
                R_cw = [[float(R_raw[r*3+c]) for c in range(3)] for r in range(3)]
            else:
                R_cw = [[float(v) for v in row] for row in R_raw]
            t_cw = [float(v) for v in t_raw]

            # Camera centre: C = -R_wc · t_cw
            R_wc = [[R_cw[c][r] for c in range(3)] for r in range(3)]
            Cx = -sum(R_wc[0][j] * t_cw[j] for j in range(3))
            Cy = -sum(R_wc[1][j] * t_cw[j] for j in range(3))
            Cz = -sum(R_wc[2][j] * t_cw[j] for j in range(3))

            # LFS position and rotation (verified formula)
            lfs_pos = (Cx, -Cy, -Cz)
            qw, qx, qy, qz = _matrix_to_quat(R_cw)
            lfs_rot = (qw, -qx, qy, qz)

            fx = float(getattr(cam, "camera_focal_x", 50.0))
            w  = int(getattr(cam, "camera_width", 1))
            focal_mm = round(_focal_to_mm(fx, w), 4)

            # ── Read world_transform and apply to COLMAP centre, then flip ────
            # Apply W3 (as stored) * C_colmap + col3, then do the LFS Y/Z flip.
            try:
                wt = cam.world_transform
                if hasattr(wt, "tolist"):
                    wt = wt.tolist()
                W  = [[float(wt[r][c]) for c in range(4)] for r in range(4)]
                W3 = [[W[r][c] for c in range(3)] for r in range(3)]
                col3 = [W[r][3] for r in range(3)]
            except Exception:
                W3   = [[1,0,0],[0,1,0],[0,0,1]]
                col3 = [0.0, 0.0, 0.0]

            Cx_w = W3[0][0]*Cx + W3[0][1]*Cy + W3[0][2]*Cz + col3[0]
            Cy_w = W3[1][0]*Cx + W3[1][1]*Cy + W3[1][2]*Cz + col3[1]
            Cz_w = W3[2][0]*Cx + W3[2][1]*Cy + W3[2][2]*Cz + col3[2]
            lfs_pos = (Cx_w, -Cy_w, -Cz_w)

            # ── Rotation: apply W3 to R_lfs ───────────────────────────────────
            qw, qx, qy, qz = lfs_rot
            R_lfs = _quat_to_matrix(qw, qx, qy, qz)
            R_final = [[sum(W3[r][k]*R_lfs[k][c] for k in range(3))
                        for c in range(3)] for r in range(3)]
            lfs_rot = _matrix_to_quat(R_final)

            result.append({"name": name, "position": lfs_pos,
                           "rotation": lfs_rot, "focal_mm": focal_mm})
        except Exception as e:
            lf.log.warn(f"ColmapCamPath: skipping camera {name!r} — {e}")

    if skipped:
        lf.log.info(f"ColmapCamPath: skipped {skipped} training-disabled camera(s)")

    result.sort(key=lambda c: _natural_key(c["name"]))
    return result


# ── Build path(s) ────────────────────────────────────────────────────────────

def build_paths(fps: float, skip_disabled: bool, max_keyframes: int) -> list:
    """Build one or more LFS camera-path dicts from scene cameras.

    If max_keyframes is 0/1/None or >= the camera count, returns a single
    dict with every camera. Otherwise the cameras are split into
    consecutive chunks of at most max_keyframes each (in natural-sort
    order), with each part after the first REPEATING the previous part's
    final camera as its own first camera -- a 1-keyframe overlap, so that
    videos rendered from each part splice together seamlessly (part 1
    ends on the same pose part 2 begins on, etc.) rather than skipping or
    duplicating motion at the join. e.g. with max_keyframes=200: part 1 is
    cameras 1-200, part 2 is cameras 200-400, part 3 is cameras 400-600.

    Each chunk's keyframe times restart at 0, so it plays back as a
    self-contained clip when loaded/rendered on its own.
    """
    cameras = _get_scene_cameras(skip_disabled)
    if not cameras:
        raise ValueError("No cameras to export (all may be disabled for training).")

    n = len(cameras)
    if not max_keyframes or max_keyframes <= 1 or max_keyframes >= n:
        chunks = [cameras]
    else:
        num_logical_parts = math.ceil(n / max_keyframes)
        chunks = []
        for i in range(num_logical_parts):
            base_start = i * max_keyframes
            base_end = min((i + 1) * max_keyframes, n)
            start = base_start - 1 if i > 0 else base_start  # repeat previous part's last camera
            chunks.append(cameras[start:base_end])

    results = []
    for chunk in chunks:
        keyframes = []
        for idx, cam in enumerate(chunk):
            qw, qx, qy, qz = cam["rotation"]
            pos = cam["position"]
            keyframes.append({
                "easing": 0,
                "focal_length_mm": cam["focal_mm"],
                "position": [round(pos[0], 6), round(pos[1], 6), round(pos[2], 6)],
                "rotation": [round(qw, 6), round(qx, 6), round(qy, 6), round(qz, 6)],
                "time": round(idx / fps, 6),
            })
        clip_duration = max((kf["time"] for kf in keyframes), default=0.0)
        results.append({"keyframes": keyframes, "version": 4, "clip_duration": clip_duration})
    return results


def part_output_path(base: Path, part_idx: int, num_parts: int) -> Path:
    """Filename for part `part_idx` (1-based) of `num_parts`. Returns
    `base` unchanged when num_parts <= 1 (no splitting needed); otherwise
    appends a zero-padded _PtNN suffix before the extension, e.g.
    colmap_camera_path.json -> colmap_camera_path_Pt01.json."""
    if num_parts <= 1:
        return base
    stem = base.stem
    suffix = base.suffix or ".json"
    width = max(2, len(str(num_parts)))
    return base.with_name(f"{stem}_Pt{part_idx:0{width}d}{suffix}")


# ── Panel ─────────────────────────────────────────────────────────────────────

class MainPanel(lf.ui.Panel):
    id          = "ColmapCamPath.main_panel"
    label       = "Cam2Seq"
    space       = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order       = 110
    template    = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT

    def __init__(self):
        self._fps           = 2.0
        self._fps_text      = "2"
        self._skip_disabled = True
        self._output_path   = str(_SCRIPTS_DIR / "colmap_camera_path.json")

        self._max_keyframes      = 400.0
        self._max_keyframes_text = "400"

        self._part      = 1.0   # 1-based index of the currently selected part
        self._part_text = "1"
        self._num_parts = 1     # how many part files the last Build produced

        self._cam_count_text   = ""
        self._parts_info_text  = ""
        self._status           = ""
        self._status_ok        = True
        self._last_backup_text = ""

        self._sec_options   = True
        self._sec_output    = True
        self._sec_sequencer = True

        self._handle = None

    # ── Model ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(v, is_int):
        if is_int:
            return str(int(round(v)))
        s = f"{round(v, 4):.4f}".rstrip("0").rstrip(".")
        return s or "0"

    def _bind_numeric(self, model, name, is_int):
        float_attr = f"_{name}"
        text_attr  = f"_{name}_text"

        def sg(fa=float_attr):          return getattr(self, fa)
        def ss(v, fa=float_attr, ta=text_attr, ii=is_int, nm=name):
            try: fv = float(int(round(float(v)))) if ii else float(v)
            except (TypeError, ValueError): return
            setattr(self, fa, fv)
            setattr(self, ta, self._fmt(fv, ii))
            if self._handle: self._handle.dirty(f"{nm}_text")
        model.bind(name, sg, ss)

        def tg(ta=text_attr):           return getattr(self, ta)
        def ts(v, fa=float_attr, ta=text_attr, ii=is_int, nm=name):
            setattr(self, ta, str(v))
            try: fv = float(int(round(float(v)))) if ii else float(v)
            except (TypeError, ValueError): return
            setattr(self, fa, fv)
            if self._handle: self._handle.dirty(nm)
        model.bind(f"{name}_text", tg, ts)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model(MODEL_NAME)
        if model is None:
            return

        self._bind_numeric(model, "fps", is_int=False)
        self._bind_numeric(model, "max_keyframes", is_int=True)

        # Part selector: clamped to [1, num_parts] on every set, rather than
        # a plain numeric bind, since num_parts changes after each Build and
        # the buttons' step logic needs the same clamped-set behaviour.
        def get_part(): return self._part
        def set_part(v):
            try:
                iv = int(round(float(v)))
            except (TypeError, ValueError):
                return
            self._set_part(iv)
        model.bind("part", get_part, set_part)

        def get_part_text(): return self._part_text
        def set_part_text(v):
            try:
                iv = int(round(float(v)))
            except (TypeError, ValueError):
                self._part_text = str(v)  # let them keep typing; don't clamp mid-edit
                if self._handle:
                    self._handle.dirty("part_text")
                return
            self._set_part(iv)
        model.bind("part_text", get_part_text, set_part_text)

        model.bind_func("num_parts", lambda: self._num_parts)
        model.bind_func("parts_info_text", lambda: self._parts_info_text)

        def get_skip():  return self._skip_disabled
        def set_skip(v): self._skip_disabled = bool(v)
        model.bind("skip_disabled", get_skip, set_skip)

        def get_sec_options():  return self._sec_options
        def set_sec_options(v): self._sec_options = bool(v)
        model.bind("sec_options", get_sec_options, set_sec_options)

        def get_sec_output():  return self._sec_output
        def set_sec_output(v): self._sec_output = bool(v)
        model.bind("sec_output", get_sec_output, set_sec_output)

        def get_sec_seq():  return self._sec_sequencer
        def set_sec_seq(v): self._sec_sequencer = bool(v)
        model.bind("sec_sequencer", get_sec_seq, set_sec_seq)

        def get_out():  return self._output_path
        def set_out(v): self._output_path = str(v)
        model.bind("output_path", get_out, set_out)

        model.bind_func("cam_count_text",   lambda: self._cam_count_text)
        model.bind_func("status",           lambda: self._status)
        model.bind_func("status_ok",        lambda: self._status_ok)
        model.bind_func("last_backup_text", lambda: self._last_backup_text)

        self._handle = model.get_handle()

    def _set_status(self, msg, ok=True):
        self._status    = msg
        self._status_ok = ok
        if self._handle:
            self._handle.dirty_all()

    def _refresh_parts_info_text(self):
        if self._num_parts <= 1:
            self._parts_info_text = ""
            return
        base = Path(self._output_path.strip() or "colmap_camera_path.json")
        target = part_output_path(base, int(self._part), self._num_parts)
        self._parts_info_text = f"Part {int(self._part)} of {self._num_parts} — {target.name}"

    def _set_part(self, iv: int):
        """Clamp to [1, num_parts], update part/part_text/parts_info_text,
        and push a redraw. Shared by the part field, and the prev/next
        step buttons."""
        iv = max(1, min(int(iv), max(1, self._num_parts)))
        self._part = float(iv)
        self._part_text = str(iv)
        self._refresh_parts_info_text()
        if self._handle:
            self._handle.dirty_all()

    def _do_part_prev(self):
        self._set_part(int(round(self._part)) - 1)

    def _do_part_next(self):
        self._set_part(int(round(self._part)) + 1)

    # ── DOM ───────────────────────────────────────────────────────────────────

    def on_mount(self, doc):
        for btn_id, fn in [
            ("btn-refresh",  self._do_refresh),
            ("btn-build",    self._do_build),
            ("btn-load",     self._do_load),
            ("btn-backup",   self._do_backup),
            ("btn-part-prev", self._do_part_prev),
            ("btn-part-next", self._do_part_next),
        ]:
            el = doc.get_element_by_id(btn_id)
            if el:
                el.add_event_listener("click", lambda _ev, f=fn: f())

        try:
            for el in doc.query_selector_all(".field-input"):
                el.add_event_listener("focus", lambda _ev, e=el: e.select() if hasattr(e, "select") else None)
        except Exception:
            pass

    # ── Actions ───────────────────────────────────────────────────────────────

    def _do_refresh(self):
        try:
            cams = _get_scene_cameras(skip_disabled=False)
            total = len(cams)
            enabled = sum(1 for c in cams if True)   # count just for display
            try:
                en_set = _training_enabled_set()
                enabled = len(en_set) if en_set is not None else total
            except Exception:
                enabled = total
            self._cam_count_text = (
                f"{total} camera(s) in scene  |  {enabled} enabled for training"
                if enabled != total else
                f"{total} camera(s) in scene"
            )
        except Exception as e:
            self._cam_count_text = f"Error: {e}"
        if self._handle:
            self._handle.dirty("cam_count_text")
        self._set_status("✓ Camera count refreshed.")

    def _do_build(self):
        out = self._output_path.strip()
        if not out:
            self._set_status("⚠ No output path set.", False); return

        try:
            max_kf = int(round(self._max_keyframes))
        except (TypeError, ValueError):
            max_kf = 0

        try:
            path_dicts = build_paths(fps=self._fps, skip_disabled=self._skip_disabled, max_keyframes=max_kf)
            base = Path(out)
            base.parent.mkdir(parents=True, exist_ok=True)
            num_parts = len(path_dicts)
            written = []
            for i, data in enumerate(path_dicts, start=1):
                target = part_output_path(base, i, num_parts)
                with open(target, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                written.append((target, len(data["keyframes"])))
        except Exception as e:
            self._set_status(f"⚠ Build failed: {e}", False)
            lf.log.error(f"ColmapCamPath: build failed — {e}")
            return

        self._num_parts = num_parts
        # Keep the current part selection if it's still in range, else reset to 1.
        self._part = float(min(max(1, int(round(self._part))), num_parts))
        self._part_text = str(int(self._part))

        total_kf = sum(n for _, n in written)
        self._refresh_parts_info_text()
        if num_parts <= 1:
            self._set_status(f"✓ Built {total_kf} keyframes → {written[0][0]}")
        else:
            self._set_status(f"✓ Built {num_parts} parts ({total_kf} keyframes total, max {max_kf}/part) → {base.parent}")

        if self._handle:
            self._handle.dirty_all()

        lf.log.info(f"ColmapCamPath: wrote {num_parts} part(s), {total_kf} keyframes total")

    def _do_load(self):
        out = self._output_path.strip()
        if not out:
            self._set_status("⚠ No output path set.", False); return

        base = Path(out)
        target = part_output_path(base, int(self._part), self._num_parts)
        if not target.exists():
            self._set_status(f"⚠ Nothing to load — run Build first ({target.name} not found).", False)
            return

        self._write_backup()
        try:
            lf.ui.load_camera_path(str(target))
            if self._num_parts > 1:
                self._set_status(f"✓ Loaded Part {int(self._part)} of {self._num_parts}: {target}")
            else:
                self._set_status(f"✓ Loaded into Sequencer: {target}")
        except Exception as e:
            self._set_status(f"⚠ Load failed: {e}", False)

    def _do_backup(self):
        p = self._write_backup()
        if p:
            self._set_status(f"✓ Backed up → {p.name}")
        else:
            self._set_status("⚠ Backup failed — is there a path in the Sequencer?", False)

    def _write_backup(self):
        import datetime
        try:
            _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            p = _BACKUP_DIR / f"backup_{stamp}.json"
            ok = lf.ui.save_camera_path(str(p))
            if not ok:
                return None
            self._last_backup_text = f"Last backup: {p.name}"
            if self._handle:
                self._handle.dirty("last_backup_text")
            return p
        except Exception as e:
            lf.log.warn(f"ColmapCamPath: backup failed — {e}")
            return None
