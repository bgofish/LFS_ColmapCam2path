# ColmapCamPath
<img width="2333" height="1515" alt="image" src="https://github.com/user-attachments/assets/597e2bcb-a3cc-485e-bb36-e5c741031919" />

Converts Scene cameras into a LichtFeld Studio camera-path JSON, and manual button to load it into the
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

## Updates
0.2.5 **Improved Sliders & Defaults**: Changed to appearance & function of the sliders and lowered the defaults

0.2.4 **Ease in/out**: off by default (every keyframe eases at constant speed, `easing: 0`). When checked, each part gets its own ease in/out

0.2.3  To avoid crashes with high camera counts the json files can be split into parts of a set number of keyframes & loaded one at at time manually.  If you request 50 frames as the break - then 51 are created so a video created from each part can be spliced together

0.2.2 when camera/point cloud is scaled/rotated/Translated then the created seq.keyframes was out of sync. This is now corrected by applying the same world transform to the seq.keyframes.



