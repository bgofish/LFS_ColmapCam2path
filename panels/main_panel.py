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

def _debug_scene_tree() -> str:
    """Log the first camera's raw data and show exactly what our pipeline
    computes at each step.  Run this AFTER applying scene rotation/translation
    so we can verify the world_transform composition."""
    lines = []
    try:
        scene = lf.get_scene()
        cams  = scene.get_active_cameras() if scene is not None else []
        if not cams:
            return "No cameras found."

        cam  = cams[0]
        name = getattr(cam, "name", "?")
        lines.append(f"=== Camera: {name} ===")

        # ── Raw COLMAP values ─────────────────────────────────────────────────
        R_raw = _tensor_to_list(cam.camera_R)
        t_raw = _tensor_to_list(cam.camera_T)
        if R_raw and not isinstance(R_raw[0], (list, tuple)):
            R_cw = [[float(R_raw[r*3+c]) for c in range(3)] for r in range(3)]
        else:
            R_cw = [[float(v) for v in row] for row in R_raw]
        t_cw = [float(v) for v in t_raw]
        lines.append(f"camera_R (R_cw): {R_cw}")
        lines.append(f"camera_T (t_cw): {t_cw}")

        # ── Step 1: COLMAP centre ─────────────────────────────────────────────
        R_wc = [[R_cw[c][r] for c in range(3)] for r in range(3)]
        Cx = -sum(R_wc[0][j]*t_cw[j] for j in range(3))
        Cy = -sum(R_wc[1][j]*t_cw[j] for j in range(3))
        Cz = -sum(R_wc[2][j]*t_cw[j] for j in range(3))
        lines.append(f"Step1 COLMAP centre C: ({Cx:.4f}, {Cy:.4f}, {Cz:.4f})")

        # ── Step 2: LFS flip ──────────────────────────────────────────────────
        lfs_pos = (Cx, -Cy, -Cz)
        qw, qx, qy, qz = _matrix_to_quat(R_cw)
        lfs_rot = (qw, -qx, qy, qz)
        lines.append(f"Step2 LFS pos (no scene xform): {[round(v,4) for v in lfs_pos]}")
        lines.append(f"Step2 LFS rot (no scene xform): {[round(v,6) for v in lfs_rot]}")

        # ── Step 3: world_transform ───────────────────────────────────────────
        try:
            wt_raw = cam.world_transform
            if hasattr(wt_raw, "tolist"):
                wt_raw = wt_raw.tolist()
            W = [[float(wt_raw[r][c]) for c in range(4)] for r in range(4)]
            lines.append(f"world_transform row0: {[round(v,6) for v in W[0]]}")
            lines.append(f"world_transform row1: {[round(v,6) for v in W[1]]}")
            lines.append(f"world_transform row2: {[round(v,6) for v in W[2]]}")
            lines.append(f"world_transform row3: {[round(v,6) for v in W[3]]}")

            W3 = [[W[r][c] for c in range(3)] for r in range(3)]
            Wt_col = [W[r][3] for r in range(3)]   # translation in column 3
            Wt_row = [W[3][c] for c in range(3)]   # translation in row 3
            lines.append(f"Translation (col3): {[round(v,4) for v in Wt_col]}")
            lines.append(f"Translation (row3): {[round(v,4) for v in Wt_row]}")

            # Apply and log result both ways so we can see which is right
            px, py, pz = lfs_pos
            pos_col3 = (
                W3[0][0]*px + W3[0][1]*py + W3[0][2]*pz + Wt_col[0],
                W3[1][0]*px + W3[1][1]*py + W3[1][2]*pz + Wt_col[1],
                W3[2][0]*px + W3[2][1]*py + W3[2][2]*pz + Wt_col[2],
            )
            # Apply and log result
            px, py, pz = lfs_pos
            pos_final = (
                W3[0][0]*px + W3[0][1]*py + W3[0][2]*pz + Wt_col[0],
                W3[1][0]*px + W3[1][1]*py + W3[1][2]*pz + Wt_col[1],
                W3[2][0]*px + W3[2][1]*py + W3[2][2]*pz + Wt_col[2],
            )
            lines.append(f"Step3 pos with col3 trans: {[round(v,4) for v in pos_final]}")

            # Rotation composition
            qw2, qx2, qy2, qz2 = lfs_rot
            R_lfs = _quat_to_matrix(qw2, qx2, qy2, qz2)
            R_final = [[sum(W3[r][k]*R_lfs[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
            rot_final = _matrix_to_quat(R_final)
            lines.append(f"Step3 rot final: {[round(v,6) for v in rot_final]}")
            lines.append(f"KEYFRAME WILL BE: pos={[round(v,4) for v in pos_final]} rot={[round(v,6) for v in rot_final]}")

            lines.append(f"world_transform col3 as position: {Wt_col}")
            lines.append(f"world_transform row3 as position: {Wt_row}")

        except Exception as e:
            lines.append(f"world_transform read failed: {e}")

        # ── All scene nodes with their transforms ─────────────────────────────
        lines.append("--- All nodes ---")
        for node in scene.get_nodes():
            nname = getattr(node, "name", "?")
            ntype = str(getattr(node, "type", "?"))
            try:
                matrix = lf.get_node_transform(nname)
                d = lf.decompose_transform(matrix)
                t = d.get("translation", [0,0,0])
                r = d.get("rotation_euler_deg", [0,0,0])
                s = d.get("scale", [1,1,1])
                if any(abs(v) > 0.001 for v in list(t)+list(r)+[s[0]-1,s[1]-1,s[2]-1]):
                    lines.append(f"  NON-IDENTITY [{ntype}] {nname!r}: t={[round(v,3) for v in t]} r={[round(v,3) for v in r]} s={[round(v,4) for v in s]}")
            except Exception:
                pass

    except Exception as e:
        lines.append(f"Debug failed: {e}")

    for line in lines:
        lf.log.info(f"ColmapDbg: {line}")

    return f"Logged {len(lines)} lines — check LFS log for 'ColmapDbg:'."


    """Walk every node in the scene and log its name, type, parent, and
    raw decompose_transform() output.  Also probes the camera object for
    any transform-related attributes beyond camera_R / camera_T.
    Call this after rotating/translating the scene so we can see where
    the applied transform actually lives."""
    lines = []

    # ── 1. All scene nodes ────────────────────────────────────────────────────
    try:
        scene = lf.get_scene()
        nodes = scene.get_nodes()
        lines.append(f"get_nodes() returned {len(nodes)} node(s)")
        for node in nodes:
            name     = getattr(node, "name",   "?")
            ntype    = str(getattr(node, "type", "?"))
            parent   = getattr(node, "parent", None)
            pname    = getattr(parent, "name", str(parent)) if parent is not None else "None"

            # Decompose via by-name lookup
            try:
                matrix = lf.get_node_transform(name)
                d = lf.decompose_transform(matrix)
            except Exception as e:
                d = {"error": str(e)}

            # Also try node.transform / node.world_transform directly
            extra = {}
            for attr in ("transform", "world_transform", "local_transform",
                         "translation", "rotation", "scale", "matrix"):
                v = getattr(node, attr, None)
                if v is not None:
                    extra[attr] = str(v)

            lines.append(
                f"  [{ntype}] {name!r}  parent={pname!r}"
                f"  decompose={d}"
                + (f"  extra={extra}" if extra else "")
            )
    except Exception as e:
        lines.append(f"get_nodes() failed: {e}")

    # ── 2. Camera object extra transform attributes ───────────────────────────
    try:
        cams = scene.get_active_cameras() if scene is not None else []
        if cams:
            cam = cams[0]
            name = getattr(cam, "name", "?")
            lines.append(f"Camera object {name!r} — probing transform attrs:")
            for attr in ("transform", "world_transform", "local_transform",
                         "node_transform", "scene_transform",
                         "translation", "position", "rotation",
                         "matrix", "pose", "extrinsic"):
                try:
                    v = getattr(cam, attr)
                    if not callable(v):
                        lines.append(f"    {attr} = {v}")
                except Exception as e:
                    pass   # silently skip missing attrs
            # Try likely method names
            for meth in ("get_transform", "get_world_transform",
                         "get_node_transform", "get_pose", "get_extrinsic"):
                fn = getattr(cam, meth, None)
                if callable(fn):
                    try:
                        lines.append(f"    {meth}() = {fn()}")
                    except Exception as e:
                        lines.append(f"    {meth}() raised: {e}")
    except Exception as e:
        lines.append(f"Camera object probe failed: {e}")

    for line in lines:
        lf.log.info(f"ColmapCamPath tree — {line}")

    return f"Logged {len(lines)} line(s) — check the LFS log."


    """Return a set of camera names that are enabled for training.
    Uses get_nodes() + node.training_enabled (same API as Cameras plugin)."""
    try:
        scene = lf.get_scene()
        return {
            node.name
            for node in scene.get_nodes()
            if str(node.type) == "NodeType.CAMERA" and node.training_enabled
        }
    except Exception as e:
        lf.log.warn(f"ColmapCamPath: could not read training states — {e}")
        return None   # None = unknown, caller treats all as enabled


# ── Scene camera reading ──────────────────────────────────────────────────────

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

            # ── Read world_transform (scene-level rotation/translation/scale) ──
            # world_transform is a 4x4 row-major matrix on the camera object,
            # expressed in LFS space (after the COLMAP→LFS flip already applied).
            # Compose it with the converted position and rotation so that any
            # scene-level transform (rotate/translate/scale the camera group)
            # is baked into the keyframe.
            try:
                wt = cam.world_transform   # tuple-of-tuples 4x4
                # Flatten to list-of-lists
                if hasattr(wt, "tolist"):
                    wt = wt.tolist()
                W  = [[float(wt[r][c]) for c in range(4)] for r in range(4)]
                W3 = [[W[r][c] for c in range(3)] for r in range(3)]
                Wt = [W[r][3] for r in range(3)]
                has_world = True
            except Exception:
                W3 = [[1,0,0],[0,1,0],[0,0,1]]
                Wt = [0.0, 0.0, 0.0]
                has_world = False

            # ── Apply world transform to position ────────────────────────────
            # p_final = W3 · p_lfs + Wt
            px, py, pz = lfs_pos
            lfs_pos = (
                W3[0][0]*px + W3[0][1]*py + W3[0][2]*pz + Wt[0],
                W3[1][0]*px + W3[1][1]*py + W3[1][2]*pz + Wt[1],
                W3[2][0]*px + W3[2][1]*py + W3[2][2]*pz + Wt[2],
            )

            # ── Apply world transform to rotation ────────────────────────────
            # R_final = W3 · R_lfs  (both expressed in LFS space)
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


# ── Build path ────────────────────────────────────────────────────────────────

def build_path(fps: float, skip_disabled: bool) -> dict:
    cameras = _get_scene_cameras(skip_disabled)
    if not cameras:
        raise ValueError("No cameras to export (all may be disabled for training).")

    keyframes = []
    for idx, cam in enumerate(cameras):
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
    return {"keyframes": keyframes, "version": 3}


# ── Panel ─────────────────────────────────────────────────────────────────────

class MainPanel(lf.ui.Panel):
    id          = "ColmapCamPath.main_panel"
    label       = "COLMAP to LFS"
    space       = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order       = 110
    template    = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT

    def __init__(self):
        self._fps           = 24.0
        self._fps_text      = "24"
        self._skip_disabled = True
        self._output_path   = str(_SCRIPTS_DIR / "colmap_camera_path.json")

        self._cam_count_text   = ""
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

    # ── DOM ───────────────────────────────────────────────────────────────────

    def on_mount(self, doc):
        for btn_id, fn in [
            ("btn-refresh",  self._do_refresh),
            ("btn-build",    self._do_build),
            ("btn-load",     self._do_load),
            ("btn-backup",   self._do_backup),
            ("btn-debug",    self._do_debug),
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
            data = build_path(fps=self._fps, skip_disabled=self._skip_disabled)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._set_status(f"⚠ Build failed: {e}", False)
            lf.log.error(f"ColmapCamPath: build failed — {e}")
            return
        n = len(data["keyframes"])
        self._set_status(f"✓ Built {n} keyframes → {out}")
        lf.log.info(f"ColmapCamPath: wrote {n} keyframes to {out!r}")

    def _do_load(self):
        out = self._output_path.strip()
        if not out or not Path(out).exists():
            self._set_status("⚠ Nothing to load — run Build first.", False); return
        self._write_backup()
        try:
            lf.ui.load_camera_path(out)
            self._set_status(f"✓ Loaded into Sequencer: {out}")
        except Exception as e:
            self._set_status(f"⚠ Load failed: {e}", False)

    def _do_debug(self):
        try:
            msg = _debug_scene_tree()
        except Exception as e:
            msg = f"Debug failed: {e}"
        self._set_status(msg)

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
