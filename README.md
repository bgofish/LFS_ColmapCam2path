# ColmapCamPath
<img width="1813" height="938" alt="image" src="https://github.com/user-attachments/assets/a4c09e7d-7c43-458f-bb1f-882743efeab5" />

Converts Scene cameras into a LichtFeld Studio camera-path JSON, and loads it straight into the
Sequencer.

## Use

1. **Conversion Options**: adjust FPS and select [Skip disabled cameras] if required
2. **Output**: set the output `.json` path (defaults to this plugin's
   Scripts folder) and click **Build Path → JSON**.
4. **Sequencer**:
   - **Load into Sequencer** automatically backs up whatever camera path is
     currently in the Sequencer first, then loads the converted one in —
     so an import never causes loss of existing work.
   - **Backup Sequencer Now** backs up the current path on demand, any
     time, independent of loading.

Backups are written as timestamped files under
`~/.lichtfeld/plugins/ColmapCamPath/Scripts/backups/`, so nothing is ever
silently overwritten.

## Known issues

1. If camera/point cloud is scaled/rotated/Translated then the created seq.keyframes are out of sync. This can be corrected by applying the same world transform to the seq.keyframes

