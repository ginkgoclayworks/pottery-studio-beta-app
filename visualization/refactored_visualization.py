#!/usr/bin/env python3
"""
Chart generation and figure management for visualization.
Handles matplotlib figure capture, safety patches, and chart creation.
"""

import io
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple


class FigureCapture:
    """
    Context manager for capturing matplotlib figures during simulation runs.
    Redirects plt.show() to save numbered PNGs and maintains a manifest.
    """
    
    def __init__(self, title_suffix: str = ""):
        self.title_suffix = title_suffix
        self._orig_show = None
        self.images: List[Tuple[str, bytes]] = []
        self.manifest = []

    def __enter__(self):
        matplotlib.use("Agg", force=True)
        self._orig_show = plt.show
        counter = {"i": 0}

        def _title_for(fig):
            """Extract title from figure or axes."""
            parts = []
            if fig._suptitle:
                txt = fig._suptitle.get_text()
                if txt:
                    parts.append(txt)
            for ax in fig.axes:
                t = getattr(ax, "get_title", lambda: "")()
                if t:
                    parts.append(t)
            return " | ".join(parts).strip()

        def _ensure_suffix(fig):
            """Add title suffix to figure if none exists."""
            if not self.title_suffix:
                return
            has_any_title = any(ax.get_title() for ax in fig.get_axes())
            if not has_any_title:
                fig.suptitle(self.title_suffix)

        def _show(*args, **kwargs):
            """Replacement for plt.show() that saves figures."""
            counter["i"] += 1
            fig = plt.gcf()
        
            _ensure_suffix(fig)
        
            has_suptitle = bool(fig._suptitle and fig._suptitle.get_text())
            has_ax_titles = any(ax.get_title() for ax in fig.get_axes())
        
            if has_suptitle and has_ax_titles:
                fig._suptitle.set_y(0.98)
                try:
                    fig._suptitle.set_fontsize(max(fig._suptitle.get_fontsize() - 2, 10))
                except Exception:
                    pass
                fig.tight_layout(rect=[0, 0, 1, 0.94])
            else:
                fig.tight_layout()
        
            buf = io.BytesIO()
            fig.savefig(buf, dpi=200, bbox_inches="tight", format="png")
            buf.seek(0)
            fname = f"fig_{counter['i']:02d}.png"
            self.images.append((fname, buf.read()))
            self.manifest.append({"file": fname, "title": _title_for(fig)})
            plt.close(fig)

        plt.show = _show
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_show:
            plt.show = self._orig_show


def create_safe_heatmap(data, **kwargs):
    """
    Safely create heatmap, handling empty or invalid data.
    
    Args:
        data: Data for heatmap
        **kwargs: Additional arguments for seaborn.heatmap
    
    Returns:
        matplotlib axes object
    """
    import numpy as np
    import seaborn as sns
    
    # Convert to numpy array if it's a DataFrame
    if hasattr(data, 'values'):
        data_array = data.values
    else:
        data_array = np.array(data)
    
    # Check if data is valid for heatmap
    if data_array.size == 0:
        print("WARNING: Empty data for heatmap, skipping...")
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'No data available for heatmap', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(kwargs.get('title', 'Empty Heatmap'))
        return plt.gca()
    
    # Check for all NaN values
    if np.isnan(data_array).all():
        print("WARNING: All NaN data for heatmap, skipping...")
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'All data is NaN', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(kwargs.get('title', 'Invalid Heatmap Data'))
        return plt.gca()
    
    # Check for valid numeric range
    valid_data = data_array[~np.isnan(data_array)]
    if len(valid_data) == 0:
        print("WARNING: No valid numeric data for heatmap, skipping...")
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'No valid numeric data', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(kwargs.get('title', 'No Valid Data'))
        return plt.gca()
    
    # If we have valid data, create the heatmap
    try:
        return sns.heatmap(data, **kwargs)
    except Exception as e:
        print(f"ERROR creating heatmap: {e}")
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f'Heatmap error: {str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(kwargs.get('title', 'Heatmap Error'))
        return plt.gca()


def apply_visualization_patches():
    """
    Apply safety patches to prevent visualization crashes.
    Should be called before running simulations.
    """
    import seaborn as sns
    import numpy as np
    
    # Store original heatmap function
    if not hasattr(sns, '_original_heatmap'):
        sns._original_heatmap = sns.heatmap
    
    def safe_heatmap_wrapper(data, **kwargs):
        """Safe wrapper around seaborn.heatmap"""
        # Quick validation
        if hasattr(data, 'values'):
            data_array = data.values
        else:
            data_array = np.array(data)
        
        if data_array.size == 0 or np.isnan(data_array).all():
            print("Skipping invalid heatmap data")
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'Invalid heatmap data', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            return plt.gca()
        
        return sns._original_heatmap(data, **kwargs)
    
    # Apply the patch
    sns.heatmap = safe_heatmap_wrapper
