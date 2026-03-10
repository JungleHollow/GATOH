from __future__ import annotations

import numpy as np
import rustworkx as rx
from matplotlib import pyplot as plt
from rustworkx.visualization import mpl_draw

from .model import ABModel


class ABVisualiser:
    """
    The visualisations will include:
        - Graph structure plotting
        - Realtime model runtime display
        - Post-runtime summarisation graphs
        - Per-agent lifetime information
        - Per-graph lifetime information
    """

    def __init__(self, model: ABModel) -> None:
        self.parent_model: ABModel = model
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
