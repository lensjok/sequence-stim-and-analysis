# -*- coding: utf-8 -*-
"""
roi_response_map.py

Colour the Suite2p ROIs by their mean response, viewed in napari.

Three map types:
    'amp'    continuous - mean dF/F for one cue (whole sequence, or one
             segment). Diverging colormap centred on zero.
    'pref'   categorical - which segment drove each cell hardest. This is
             the retinotopy map: if the sequence works, 'pref' for
             cue1_azimuth should form a left-to-right gradient across the
             field of view.
    'latency' continuous - time of peak response. Same information as
             'pref' but continuous, and often cleaner to look at.

Everything below the napari import is a pure function, so the maps can be
computed and saved headlessly too.

Integration with the existing GUI - three lines in FirstLookGUI._build_ui:

    from roi_response_map import ResponseMapWidget
    self.map_widget = ResponseMapWidget(self)
    layout.addWidget(self.map_widget)

ResponseMapWidget expects the parent to expose .viewer, .stat, .ops,
.cell_indices and .seq_results (the dict returned by sequence_analysis.analyse).
"""

from __future__ import division
import numpy as np


# ---------------------------------------------------------------------
# Pure image building
# ---------------------------------------------------------------------
def roi_value_image(stat, values, Ly, Lx, cell_indices=None, fill=np.nan,
                    weight_by_lam=False):
    """Paint one scalar per cell onto its Suite2p pixels.

    stat          the Suite2p stat array (all ROIs)
    values        (n_cells,) one value per *filtered* cell
    cell_indices  original stat indices of those cells (from iscell filtering).
                  None means values line up with stat directly.
    fill          value for pixels belonging to no ROI (NaN renders as
                  transparent-ish in napari and is ignored by contrast limits)

    Overlapping ROIs: later cells win, which matches how Suite2p draws them.
    """
    img = np.full((int(Ly), int(Lx)), fill, dtype=np.float32)
    idx = np.arange(len(stat)) if cell_indices is None else np.asarray(cell_indices)
    values = np.asarray(values, dtype=np.float32)
    if len(values) != len(idx):
        raise ValueError('values (%d) does not match cells (%d)'
                         % (len(values), len(idx)))

    for v, si in zip(values, idx):
        s = stat[si]
        ypix = np.asarray(s['ypix']).astype(int)
        xpix = np.asarray(s['xpix']).astype(int)
        ok = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        if weight_by_lam and 'lam' in s:
            lam = np.asarray(s['lam'], dtype=np.float32)
            lam = lam / (lam.max() if lam.max() > 0 else 1.0)
            img[ypix[ok], xpix[ok]] = v * lam[ok]
        else:
            img[ypix[ok], xpix[ok]] = v
    return img


def roi_label_image(stat, Ly, Lx, cell_indices=None):
    """Label image where each filtered cell gets label i+1 (0 = background)."""
    lab = np.zeros((int(Ly), int(Lx)), dtype=np.int32)
    idx = np.arange(len(stat)) if cell_indices is None else np.asarray(cell_indices)
    for i, si in enumerate(idx):
        s = stat[si]
        ypix = np.asarray(s['ypix']).astype(int)
        xpix = np.asarray(s['xpix']).astype(int)
        ok = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        lab[ypix[ok], xpix[ok]] = i + 1
    return lab


# distinct, colour-blind-safe-ish palette for up to 6 segments
SEGMENT_COLORS = [
    (0.90, 0.24, 0.24, 1.0),   # red      segment 1
    (0.99, 0.68, 0.11, 1.0),   # orange   segment 2
    (0.20, 0.63, 0.93, 1.0),   # blue     segment 3
    (0.35, 0.75, 0.35, 1.0),
    (0.72, 0.42, 0.90, 1.0),
    (0.50, 0.50, 0.50, 1.0),
]


def preferred_segment_colormap(pref_index, responsive=None, n_segments=3):
    """Build {label: RGBA} for a napari Labels layer.

    Non-responsive cells are drawn dark grey rather than hidden, so you can
    still see the ROI outlines and judge coverage.
    """
    colors = {0: (0, 0, 0, 0)}
    for i, p in enumerate(np.asarray(pref_index)):
        if responsive is not None and not responsive[i]:
            colors[i + 1] = (0.22, 0.22, 0.22, 1.0)
        else:
            colors[i + 1] = SEGMENT_COLORS[int(p) % len(SEGMENT_COLORS)]
    return colors


def symmetric_limits(values, pct=99.0):
    """Contrast limits centred on zero, robust to outliers."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    m = np.percentile(np.abs(v), pct)
    m = max(float(m), 1e-9)
    return (-m, m)


# ---------------------------------------------------------------------
# Extracting the scalar per cell
# ---------------------------------------------------------------------
def cue_amplitude(result, segment=None):
    """Mean response amplitude per cell for one cue.

    segment=None  -> mean across the whole sequence
    segment=int   -> that segment only
    segment=str   -> segment with that name
    """
    st = result['stats']
    if segment is None:
        return st['mean_amp'].mean(axis=1)
    if isinstance(segment, str):
        segment = st['names'].index(segment)
    return st['mean_amp'][:, int(segment)]


def cue_map_values(result, kind='amp', segment=None, mask_unresponsive=True):
    """Return (values, kind_info) ready for roi_value_image."""
    st = result['stats']
    if kind == 'amp':
        vals = cue_amplitude(result, segment).astype(float)
        if mask_unresponsive:
            vals = np.where(st['responsive'], vals, np.nan)
        return vals, dict(diverging=True, label='mean dF/F')
    if kind == 'latency':
        vals = result['latency']['latency_s'].astype(float)
        if mask_unresponsive:
            vals = np.where(st['responsive'], vals, np.nan)
        return vals, dict(diverging=False, label='peak latency (s)')
    raise ValueError("kind must be 'amp' or 'latency' (use 'pref' via Labels)")


# ---------------------------------------------------------------------
# napari layers
# ---------------------------------------------------------------------
def add_response_maps(viewer, stat, ops, seq_results, cell_indices=None,
                      cues=None, kinds=('amp', 'pref'), segment=None,
                      mask_unresponsive=True):
    """Add one layer per (cue, kind). Returns the created layers."""
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    cues = cues or sorted(seq_results.keys())
    layers = []

    for cue in cues:
        result = seq_results[cue]
        st = result['stats']
        tag = cue if segment is None else '%s:%s' % (cue, segment)

        if 'amp' in kinds:
            vals, info = cue_map_values(result, 'amp', segment, mask_unresponsive)
            img = roi_value_image(stat, vals, Ly, Lx, cell_indices)
            layers.append(viewer.add_image(
                img, name='dF %s' % tag, colormap='bwr',
                contrast_limits=symmetric_limits(vals), opacity=0.85,
                visible=(cue == cues[0])))

        if 'latency' in kinds:
            vals, info = cue_map_values(result, 'latency', None, mask_unresponsive)
            img = roi_value_image(stat, vals, Ly, Lx, cell_indices)
            finite = vals[np.isfinite(vals)]
            lim = (float(np.nanmin(finite)), float(np.nanmax(finite))) \
                if finite.size else (0.0, 1.0)
            layers.append(viewer.add_image(
                img, name='latency %s' % cue, colormap='turbo',
                contrast_limits=lim, opacity=0.85, visible=False))

        if 'pref' in kinds:
            lab = roi_label_image(stat, Ly, Lx, cell_indices)
            colors = preferred_segment_colormap(
                st['pref_index'],
                st['responsive'] if mask_unresponsive else None,
                len(st['names']))
            try:
                layer = viewer.add_labels(lab, name='pref segment %s' % cue,
                                          color=colors, opacity=0.85)
            except TypeError:      # napari >= 0.5 renamed the argument
                layer = viewer.add_labels(lab, name='pref segment %s' % cue,
                                          colormap=colors, opacity=0.85)
            layer.visible = False
            layers.append(layer)
            print('  %s preferred-segment colours: %s'
                  % (cue, ', '.join('%s=%s' % (nm, ['red', 'orange', 'blue',
                                                    'green', 'purple', 'grey'][i])
                                    for i, nm in enumerate(st['names']))))
    return layers


def save_maps(path_stem, stat, ops, seq_results, cell_indices=None):
    """Write the maps as .npy so they can be reloaded or figure-made later."""
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    out = {}
    for cue, result in seq_results.items():
        vals, _ = cue_map_values(result, 'amp')
        out['%s_amp' % cue] = roi_value_image(stat, vals, Ly, Lx, cell_indices)
        lat, _ = cue_map_values(result, 'latency')
        out['%s_latency' % cue] = roi_value_image(stat, lat, Ly, Lx, cell_indices)
        pref = np.where(result['stats']['responsive'],
                        result['stats']['pref_index'], -1).astype(float)
        pref[pref < 0] = np.nan
        out['%s_pref' % cue] = roi_value_image(stat, pref, Ly, Lx, cell_indices)
    np.savez_compressed(path_stem + '.npz', **out)
    return out


# ---------------------------------------------------------------------
# Optional Qt widget for the existing GUI
# ---------------------------------------------------------------------
def _make_widget_class():
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QComboBox, QCheckBox)

    class ResponseMapWidget(QWidget):
        """Dockable panel that draws the ROI response maps."""

        def __init__(self, host):
            super().__init__()
            self.host = host
            lay = QVBoxLayout(self)
            lay.addWidget(QLabel('<b>ROI response maps</b>'))

            row = QHBoxLayout()
            row.addWidget(QLabel('Cue:'))
            self.cue_box = QComboBox(); row.addWidget(self.cue_box)
            row.addWidget(QLabel('Segment:'))
            self.seg_box = QComboBox(); row.addWidget(self.seg_box)
            lay.addLayout(row)

            row2 = QHBoxLayout()
            row2.addWidget(QLabel('Map:'))
            self.kind_box = QComboBox()
            self.kind_box.addItems(['mean dF (amp)', 'preferred segment',
                                    'peak latency'])
            row2.addWidget(self.kind_box)
            self.mask_chk = QCheckBox('responsive only')
            self.mask_chk.setChecked(True)
            row2.addWidget(self.mask_chk)
            lay.addLayout(row2)

            row3 = QHBoxLayout()
            self.draw_btn = QPushButton('Draw map')
            self.draw_btn.clicked.connect(self.draw)
            row3.addWidget(self.draw_btn)
            self.save_btn = QPushButton('Save maps (.npz)')
            self.save_btn.clicked.connect(self.save)
            row3.addWidget(self.save_btn)
            lay.addLayout(row3)

            self.status = QLabel('')
            lay.addWidget(self.status)

        def refresh(self):
            res = getattr(self.host, 'seq_results', None)
            self.cue_box.clear(); self.seg_box.clear()
            if not res:
                self.status.setText('Run the sequence analysis first.')
                return
            self.cue_box.addItems(sorted(res.keys()))
            names = res[sorted(res.keys())[0]]['stats']['names']
            self.seg_box.addItems(['whole sequence'] + list(names))
            self.status.setText('%d cues loaded.' % len(res))

        def draw(self):
            res = getattr(self.host, 'seq_results', None)
            if not res:
                self.status.setText('Run the sequence analysis first.')
                return
            cue = self.cue_box.currentText()
            seg = self.seg_box.currentIndex() - 1
            seg = None if seg < 0 else seg
            k = {0: 'amp', 1: 'pref', 2: 'latency'}[self.kind_box.currentIndex()]
            add_response_maps(self.host.viewer, self.host.stat, self.host.ops,
                              {cue: res[cue]}, self.host.cell_indices,
                              cues=[cue], kinds=(k,), segment=seg,
                              mask_unresponsive=self.mask_chk.isChecked())
            for lyr in self.host.viewer.layers:
                if cue in lyr.name:
                    lyr.visible = True
            self.status.setText('Drew %s map for %s.' % (k, cue))

        def save(self):
            from qtpy.QtWidgets import QFileDialog
            res = getattr(self.host, 'seq_results', None)
            if not res:
                return
            path, _ = QFileDialog.getSaveFileName(self, 'Save maps',
                                                  'roi_maps', 'NumPy (*.npz)')
            if not path:
                return
            save_maps(path.replace('.npz', ''), self.host.stat, self.host.ops,
                      res, self.host.cell_indices)
            self.status.setText('Saved %s.npz' % path.replace('.npz', ''))

    return ResponseMapWidget


def __getattr__(name):
    if name == 'ResponseMapWidget':
        return _make_widget_class()
    raise AttributeError(name)
