"""
ColmapCamPath - A LichtFeld Studio plugin.

Converts a COLMAP sparse reconstruction (cameras.txt/.bin + images.txt/.bin)
into an LFS camera-path JSON and loads it straight into the Sequencer.
"""

import lichtfeld as lf
from .panels.main_panel import MainPanel

_classes = [MainPanel]


def on_load():
    """Called when plugin is loaded."""
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("ColmapCamPath plugin loaded")


def on_unload():
    """Called when plugin is unloaded."""
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("ColmapCamPath plugin unloaded")
