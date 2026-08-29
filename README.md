# ColmapCamPath

Converts a COLMAP sparse reconstruction (`cameras.txt`/`.bin` + `images.txt`/`.bin`)
into a LichtFeld Studio camera-path JSON, and loads it straight into the
Sequencer.

## Use

1. **COLMAP Source → Folder**: click **Browse…** and pick the COLMAP project
   folder (or paste a path directly into the field). You can point it at the
   project root, the `sparse/0` folder, or anywhere in between — subfolders
   are searched automatically (bounded depth) for `cameras.*` + `images.*`.
2. **Conversion Options**: adjust FPS, Scale, Sensor width, Roll, and an
   optional focal-length override (leave blank to derive focal length from
   the COLMAP camera intrinsics).
   - **Roll** rotates each camera about its own camera→target axis. LFS's
     camera convention needs a +90° roll relative to COLMAP's by default —
     change or zero it out if your result looks off.
3. **Output**: set the output `.json` path (defaults to this plugin's
   Scripts folder) and click **Convert COLMAP → JSON**.
4. **Sequencer**:
   - **Load into Sequencer** automatically backs up whatever camera path is
     currently in the Sequencer first, then loads the converted one in —
     so a bad conversion never destroys existing work.
   - **Backup Sequencer Now** backs up the current path on demand, any
     time, independent of loading.

Backups are written as timestamped files under
`~/.lichtfeld/plugins/ColmapCamPath/Scripts/backups/`, so nothing is ever
silently overwritten.

## Notes

- The folder browser shells out to a short PowerShell script that pops a
  native Windows Forms folder dialog (the same approach other LFS plugins
  use) — Windows-only. On macOS/Linux, just paste the folder path into the
  field directly — everything else works the same.
- Coordinate conversion (COLMAP → LFS/SuperSplat axes) and camera model
  handling (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, …) match
  the standalone `COLMAP-CAM_to_LFS-PATH-json.py` converter this plugin
  replaces.
