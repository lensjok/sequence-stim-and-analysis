#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sequence_gui.py

Standalone napari GUI for testing the 3-segment sequence analysis.
Movies are deliberately not included - use cue_average_movie.py for those.

    python sequence_gui.py

Workflow:
    1. Load Suite2p   (pick the suite2p/ or suite2p/plane0/ folder)
    2. Load behaviour (the .txt with the cue lines)
    3. check the parameters, then Run analysis
    4. Draw maps in the viewer, flip through cells, export CSV

The plotting functions at the top take a matplotlib Figure and plain arrays,
so they can be used headlessly for figures without touching Qt.

@author: analysis add-on for krisklen's pipeline
"""

from __future__ import division
import os
import sys
import numpy as np

# --- make sure the sibling modules are importable, wherever this is run from ---
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import sequence_analysis as sa
    import roi_response_map as rm
except ImportError as _exc:
    _need = ['sequence_analysis.py', 'roi_response_map.py']
    _missing = [f for f in _need if not os.path.isfile(os.path.join(_HERE, f))]
    raise SystemExit(
        "\nsequence_gui.py could not import its companion modules.\n"
        "  error   : %s\n"
        "  folder  : %s\n"
        "  missing : %s\n\n"
        "Put %s in the SAME folder as this script, then run it again.\n"
        % (_exc, _HERE, ', '.join(_missing) if _missing else 'none (import failed for another reason)',
           ' and '.join(_need)))

SEG_COLORS = ['#e63c3c', '#fdae1b', '#33a0ee', '#59bf59', '#b86be6', '#808080']


# =====================================================================
# Plotting (pure - no Qt, testable headlessly)
# =====================================================================
def _seg_dur(results, cue, fallback=0.5):
    """Segment duration measured for this cue, not an assumed 0.5 s."""
    try:
        d = float(results[cue]['stats'].get('seg_dur', fallback))
        return d if np.isfinite(d) and d > 0 else fallback
    except Exception:
        return fallback


def _shade_segments(ax, names, seg_dur=0.5, alpha=0.13):
    for k, nm in enumerate(names):
        ax.axvspan(k * seg_dur, (k + 1) * seg_dur, color=SEG_COLORS[k % 6],
                   alpha=alpha, lw=0)
    ax.axvline(0, color='k', lw=0.8, ls='-')
    ax.axvline(len(names) * seg_dur, color='k', lw=0.8, ls=':')


def plot_cell_psth(fig, results, cell, seg_dur=None, title=''):
    """Trial-averaged trace per cue for one cell, with segments shaded."""
    fig.clear()
    cues = sorted(results.keys())
    axes = fig.subplots(len(cues), 1, sharex=True, squeeze=False)[:, 0]

    for ax, cue in zip(axes, cues):
        lat = results[cue]['latency']
        st = results[cue]['stats']
        t = lat['tvec']
        y = lat['avg'][cell]
        ax.plot(t, y, color='k', lw=1.4)
        sem = lat.get('sem')
        if sem is not None:
            ax.fill_between(t, y - sem[cell], y + sem[cell], color='k', alpha=0.2, lw=0)
        _shade_segments(ax, st['names'], seg_dur or _seg_dur(results, cue))

        pk = lat['latency_s'][cell]
        ax.axvline(pk, color='#d62728', lw=1.0, ls='--')
        flag = 'RESPONSIVE' if st['responsive'][cell] else 'n.s.'
        ax.set_title('%s   %s   prefers %s   peak %.2f s   p=%.3g   '
                     '(%d x %.2f s)'
                     % (cue, flag, st['pref_name'][cell], pk, st['p_cell'][cell],
                        len(st['names']), _seg_dur(results, cue)),
                     fontsize=9)
        ax.set_ylabel('signal')
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel('time from cue onset (s)')
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_segment_tuning(fig, results, cell, seg_dur=None):
    """Bar of amplitude per segment, per cue - the retinotopy readout."""
    fig.clear()
    cues = sorted(results.keys())
    axes = fig.subplots(1, len(cues), squeeze=False)[0]
    for ax, cue in zip(axes, cues):
        st = results[cue]['stats']
        amp = st['mean_amp'][cell]
        p = st['p_seg_corr'][cell]
        cols = [SEG_COLORS[k % 6] for k in range(len(amp))]
        ax.bar(range(len(amp)), amp, color=cols,
               edgecolor='k', linewidth=0.6)
        for k, (a, pv) in enumerate(zip(amp, p)):
            if pv < 0.05:
                ax.text(k, a, '*', ha='center',
                        va='bottom' if a >= 0 else 'top', fontsize=13)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(range(len(amp)))
        ax.set_xticklabels(st['names'], rotation=30, ha='right', fontsize=8)
        ax.set_title(cue, fontsize=9)
        ax.grid(alpha=0.2, axis='y')
    axes[0].set_ylabel('amplitude (signal - baseline)')
    fig.tight_layout()
    return fig


def plot_population_sequence(fig, results, cue=None, responsive_only=True,
                             seg_dur=None, normalise=True, sort_by=None):
    """Cells x time heatmap, sorted by peak latency.

    cue=None      -> one panel per cue
    cue='name'    -> just that cue
    sort_by='name'-> use THAT cue's responsive cells and latency ordering for
                     every panel, so a row is the same cell in each. If the
                     two cues drive independent populations the second panel
                     looks unstructured; if the same cells are re-used in the
                     same order, the diagonal survives.

    If the stimulus drives a travelling wave, the bright band runs
    diagonally across the stimulus period.
    """
    fig.clear()
    cues = sorted(results.keys()) if cue in (None, 'all cues') else [cue]

    # choose the cell set and row order
    if sort_by and sort_by in results:
        ref = results[sort_by]
        sel = np.where(ref['stats']['responsive'])[0] if responsive_only \
            else np.arange(ref['latency']['avg'].shape[0])
        sel = sel[np.argsort(ref['latency']['latency_s'][sel])]
        shared = True
    else:
        shared = False

    gs = fig.add_gridspec(1, len(cues) + 1,
                          width_ratios=[1] * len(cues) + [0.05], wspace=0.12)
    im = None
    for j, cu in enumerate(cues):
        st, lat = results[cu]['stats'], results[cu]['latency']
        avg, t = lat['avg'], lat['tvec']

        if not shared:
            s_ = np.where(st['responsive'])[0] if responsive_only \
                else np.arange(avg.shape[0])
            s_ = s_[np.argsort(lat['latency_s'][s_])]
        else:
            s_ = sel

        sd = seg_dur or _seg_dur(results, cu)
        ax = fig.add_subplot(gs[j])
        if s_.size == 0:
            ax.text(0.5, 0.5, 'no responsive cells', ha='center', va='center')
            ax.set_title(cu, fontsize=10)
            continue

        m = avg[s_]
        if normalise:
            base = m[:, t < 0].mean(axis=1, keepdims=True)
            rng = np.abs(m - base).max(axis=1, keepdims=True)
            rng[rng == 0] = 1.0
            m = (m - base) / rng

        im = ax.imshow(m, aspect='auto', cmap='magma', vmin=-0.2, vmax=1.0,
                       extent=[t[0], t[-1], len(s_), 0], interpolation='nearest')
        for k in range(len(st['names']) + 1):
            ax.axvline(k * sd, color='w', lw=0.9, alpha=0.75)
        for k, nm in enumerate(st['names']):
            ax.text(k * sd + sd / 2, len(s_) * 0.025, nm,
                    ha='center', va='top', fontsize=7, color='w',
                    bbox=dict(fc=SEG_COLORS[k % 6], ec='none', alpha=0.85,
                              boxstyle='round,pad=0.16'), clip_on=True)
        ax.set_xlabel('time from cue onset (s)')
        if j == 0:
            ax.set_ylabel('cell (sorted by peak latency%s)'
                          % (' of ' + sort_by if shared else ''))
        else:
            ax.set_yticklabels([])
        ax.set_title('%s  -  %d cells' % (cu, len(s_)), fontsize=10)

    if im is not None:
        fig.colorbar(im, cax=fig.add_subplot(gs[-1]))
    return fig


def plot_latency_histogram(fig, results, seg_dur=None):
    """Distribution of peak latencies, split by preferred segment."""
    fig.clear()
    cues = sorted(results.keys())
    axes = fig.subplots(len(cues), 1, sharex=True, squeeze=False)[:, 0]
    for ax, cue in zip(axes, cues):
        st, lat = results[cue]['stats'], results[cue]['latency']
        sel = st['responsive']
        for k, nm in enumerate(st['names']):
            v = lat['latency_s'][sel & (st['pref_index'] == k)]
            if v.size:
                ax.hist(v, bins=24, alpha=0.65, label='%s (n=%d)' % (nm, v.size),
                        color=SEG_COLORS[k % 6])
        _shade_segments(ax, st['names'], seg_dur or _seg_dur(results, cue), alpha=0.08)
        ax.legend(fontsize=7)
        ax.set_title(cue, fontsize=9)
        ax.set_ylabel('cells')
    axes[-1].set_xlabel('peak latency (s)')
    fig.tight_layout()
    return fig


PLOTS = {
    'Single cell PSTH': 'psth',
    'Single cell segment tuning': 'tuning',
    'Population sequence (sorted)': 'population',
    'Peak latency histogram': 'latency',
}


# =====================================================================
# Data loading (no Qt)
# =====================================================================
def load_suite2p_min(folder, neuropil_coeff=0.7):
    """Load a Suite2p plane folder. Returns a dict."""
    if os.path.isdir(os.path.join(folder, 'plane0')):
        folder = os.path.join(folder, 'plane0')
    need = ['F.npy', 'Fneu.npy', 'iscell.npy', 'stat.npy', 'ops.npy']
    missing = [f for f in need if not os.path.isfile(os.path.join(folder, f))]
    if missing:
        raise FileNotFoundError('Missing %s in %s' % (', '.join(missing), folder))

    F = np.load(os.path.join(folder, 'F.npy'))
    Fneu = np.load(os.path.join(folder, 'Fneu.npy'))
    iscell = np.load(os.path.join(folder, 'iscell.npy'))
    stat = np.load(os.path.join(folder, 'stat.npy'), allow_pickle=True)
    ops = np.load(os.path.join(folder, 'ops.npy'), allow_pickle=True).item()

    mask = iscell[:, 0].astype(bool)
    cell_indices = np.where(mask)[0]

    Fc = F[mask] - neuropil_coeff * Fneu[mask]
    base = np.percentile(Fc, 20, axis=1, keepdims=True)
    base = np.where(np.abs(base) < 1e-6, 1.0, base)
    dff = (Fc - base) / np.abs(base)

    spks = None
    sp = os.path.join(folder, 'spks.npy')
    if os.path.isfile(sp):
        spks = np.load(sp)[mask]

    return dict(folder=folder, dff=dff, spks=spks, stat=stat, ops=ops,
                cell_indices=cell_indices, s2p_ids=cell_indices + 1,
                n_cells=int(mask.sum()), n_frames=int(F.shape[1]),
                fs=float(ops.get('fs', 30.9)))


# =====================================================================
# GUI
# =====================================================================
def build_gui():
    try:
        import napari
    except ImportError as exc:
        raise SystemExit(
            "\nCould not import napari (%s).\n"
            "You are probably in the wrong conda environment. Activate the one\n"
            "you normally run your napari GUI from, e.g.:\n\n"
            "    conda activate suite2p\n"
            "    python sequence_gui.py\n\n"
            "To check which environments exist:  conda env list\n" % exc)
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
        QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
        QGroupBox, QMessageBox)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    class MplCanvas(QWidget):
        def __init__(self, parent=None, width=6, height=5, dpi=100):
            super().__init__(parent)
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.canvas = FigureCanvasQTAgg(self.fig)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.canvas)

    class SequenceGUI(QWidget):
        def __init__(self, viewer):
            super().__init__()
            self.viewer = viewer
            self.data = None
            self.trials = None
            self.seq_results = None
            self.header = None
            self.current_cell = 0
            self._build()

            @viewer.bind_key('Left', overwrite=True)
            def _p(v):
                self.step(-1)

            @viewer.bind_key('Right', overwrite=True)
            def _n(v):
                self.step(+1)

        # ---------------- UI ----------------
        def _build(self):
            lay = QVBoxLayout(self)
            lay.addWidget(QLabel('<b>Sequence stimulus analysis (test GUI)</b>'))

            # --- loading ---
            g = QGroupBox('Data')
            gl = QVBoxLayout(g)
            r = QHBoxLayout()
            b = QPushButton('Load Suite2p'); b.clicked.connect(self.load_s2p)
            r.addWidget(b)
            self.s2p_lbl = QLabel('none'); r.addWidget(self.s2p_lbl, 1)
            gl.addLayout(r)
            r = QHBoxLayout()
            b = QPushButton('Load behaviour'); b.clicked.connect(self.load_beh)
            r.addWidget(b)
            self.beh_lbl = QLabel('none'); r.addWidget(self.beh_lbl, 1)
            gl.addLayout(r)
            lay.addWidget(g)

            # --- parameters ---
            g = QGroupBox('Parameters')
            gr = QGridLayout(g)
            self.signal_box = QComboBox()
            self.signal_box.addItems(['spks (deconvolved)', 'dF/F'])
            gr.addWidget(QLabel('Signal'), 0, 0); gr.addWidget(self.signal_box, 0, 1)

            def spin(v, lo, hi, step, dec=2):
                s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
                s.setDecimals(dec); s.setValue(v); return s

            self.fs_spin = spin(30.9, 0.1, 1000, 0.1, 3)
            gr.addWidget(QLabel('fs (Hz)'), 0, 2); gr.addWidget(self.fs_spin, 0, 3)
            self.lag_spin = spin(0.10, 0.0, 2.0, 0.05)
            gr.addWidget(QLabel('lag (s)'), 1, 0); gr.addWidget(self.lag_spin, 1, 1)
            self.tail_spin = spin(0.30, 0.0, 3.0, 0.05)
            gr.addWidget(QLabel('tail (s)'), 1, 2); gr.addWidget(self.tail_spin, 1, 3)
            self.pre_spin = spin(2.0, 0.5, 10.0, 0.5, 1)
            gr.addWidget(QLabel('pre (s)'), 2, 0); gr.addWidget(self.pre_spin, 2, 1)
            self.post_spin = spin(5.0, 1.0, 20.0, 0.5, 1)
            gr.addWidget(QLabel('post (s)'), 2, 2); gr.addWidget(self.post_spin, 2, 3)
            self.base_spin = spin(1.0, 0.1, 5.0, 0.1, 1)
            gr.addWidget(QLabel('baseline (s)'), 3, 0); gr.addWidget(self.base_spin, 3, 1)
            self.alpha_spin = spin(0.05, 0.001, 0.5, 0.01, 3)
            gr.addWidget(QLabel('alpha (FDR)'), 3, 2); gr.addWidget(self.alpha_spin, 3, 3)
            lay.addWidget(g)

            r = QHBoxLayout()
            self.run_btn = QPushButton('Run analysis')
            self.run_btn.clicked.connect(self.run_analysis); r.addWidget(self.run_btn)
            b = QPushButton('Load cached'); b.clicked.connect(self.load_cached); r.addWidget(b)
            b = QPushButton('Save .npz'); b.clicked.connect(self.save_npz); r.addWidget(b)
            b = QPushButton('Export CSV'); b.clicked.connect(self.export_csv); r.addWidget(b)
            lay.addLayout(r)

            self.status = QLabel('Load Suite2p and a behaviour file.')
            self.status.setWordWrap(True)
            lay.addWidget(self.status)

            # --- maps ---
            g = QGroupBox('ROI maps in viewer')
            gl = QVBoxLayout(g)
            r = QHBoxLayout()
            r.addWidget(QLabel('Cue')); self.cue_box = QComboBox(); r.addWidget(self.cue_box, 1)
            r.addWidget(QLabel('Segment')); self.seg_box = QComboBox(); r.addWidget(self.seg_box, 1)
            gl.addLayout(r)
            r = QHBoxLayout()
            r.addWidget(QLabel('Map')); self.kind_box = QComboBox()
            self.kind_box.addItems(['mean amplitude', 'preferred segment', 'peak latency'])
            r.addWidget(self.kind_box, 1)
            self.mask_chk = QCheckBox('responsive only'); self.mask_chk.setChecked(True)
            r.addWidget(self.mask_chk)
            b = QPushButton('Draw'); b.clicked.connect(self.draw_map); r.addWidget(b)
            gl.addLayout(r)
            lay.addWidget(g)

            # --- cell navigation ---
            g = QGroupBox('Cell')
            r = QHBoxLayout(g)
            b = QPushButton('< Prev'); b.clicked.connect(lambda: self.step(-1)); r.addWidget(b)
            b = QPushButton('Next >'); b.clicked.connect(lambda: self.step(+1)); r.addWidget(b)
            r.addWidget(QLabel('s2p ID'))
            self.id_spin = QSpinBox(); self.id_spin.setRange(1, 999999); r.addWidget(self.id_spin)
            b = QPushButton('Go'); b.clicked.connect(self.goto_id); r.addWidget(b)
            self.resp_only_chk = QCheckBox('step through responsive only')
            r.addWidget(self.resp_only_chk)
            lay.addWidget(g)
            self.cell_lbl = QLabel('Cell: -')
            lay.addWidget(self.cell_lbl)

            # --- plot ---
            r = QHBoxLayout()
            r.addWidget(QLabel('Plot'))
            self.plot_box = QComboBox(); self.plot_box.addItems(list(PLOTS.keys()))
            self.plot_box.currentIndexChanged.connect(self.refresh_plot)
            r.addWidget(self.plot_box, 1)
            r.addWidget(QLabel('cue'))
            self.plot_cue_box = QComboBox()
            self.plot_cue_box.currentIndexChanged.connect(self.refresh_plot)
            r.addWidget(self.plot_cue_box, 1)
            self.lock_order_chk = QCheckBox('same cell order')
            self.lock_order_chk.setToolTip(
                'Sort every panel by the FIRST cue\'s peak latency, so a row is '
                'the same cell in both. Shows whether the two sequences re-use '
                'the same population in the same order.')
            self.lock_order_chk.stateChanged.connect(self.refresh_plot)
            r.addWidget(self.lock_order_chk)
            b = QPushButton('Save figure'); b.clicked.connect(self.save_fig); r.addWidget(b)
            lay.addLayout(r)
            self.canvas = MplCanvas(self, width=6, height=5)
            lay.addWidget(self.canvas, 1)

        # ---------------- helpers ----------------
        def say(self, msg, colour='black'):
            self.status.setText(msg)
            self.status.setStyleSheet('color: %s;' % colour)
            print(msg); sys.stdout.flush()

        def signal(self):
            if self.signal_box.currentIndex() == 0 and self.data['spks'] is not None:
                return self.data['spks'], 'spks'
            return self.data['dff'], 'dff'

        # ---------------- actions ----------------
        def load_s2p(self):
            folder = QFileDialog.getExistingDirectory(self, 'Suite2p folder')
            if not folder:
                return
            try:
                self.data = load_suite2p_min(folder)
            except Exception as exc:
                QMessageBox.critical(self, 'Load failed', str(exc)); return
            d = self.data
            self.s2p_lbl.setText('%d cells, %d frames' % (d['n_cells'], d['n_frames']))
            self.fs_spin.setValue(d['fs'])
            if d['spks'] is None:
                self.signal_box.setCurrentIndex(1)
                self.signal_box.setItemText(0, 'spks (not found)')

            self.viewer.layers.clear()
            mean_img = d['ops'].get('meanImg')
            if mean_img is not None:
                self.viewer.add_image(np.asarray(mean_img), name='meanImg',
                                      colormap='gray')
            lab = rm.roi_label_image(d['stat'], d['ops']['Ly'], d['ops']['Lx'],
                                     d['cell_indices'])
            self.labels_layer = self.viewer.add_labels(lab, name='ROIs', opacity=0.4)
            self._hook_click(self.labels_layer)
            self.say('Loaded %s' % d['folder'], 'green')

        def _hook_click(self, layer):
            try:
                @layer.mouse_drag_callbacks.append
                def _click(lyr, event):
                    try:
                        val = lyr.get_value(event.position, world=True)
                    except TypeError:
                        val = lyr.get_value(event.position)
                    if val:
                        self.set_cell(int(val) - 1)
            except Exception as exc:
                print('click-to-select unavailable: %s' % exc)

        def load_beh(self):
            path, _ = QFileDialog.getOpenFileName(
                self, 'Behaviour file', '', 'Text (*.txt *.csv *.dat);;All (*)')
            if not path:
                return
            nf = self.data['n_frames'] if self.data else None
            try:
                self.trials, _ = sa.load_stim_trials(path, n_frames_expected=nf)
            except Exception as exc:
                QMessageBox.critical(self, 'Load failed', str(exc)); return
            self.beh_path = path
            cues = sorted(set(t['cue'] for t in self.trials))
            src = set(t['seg_source'] for t in self.trials)
            self.beh_lbl.setText('%d trials, %s' % (len(self.trials), ', '.join(cues)))
            self.say('Segment boundaries from: %s. See console for per-cue detail.'
                     % ', '.join(src), 'green')

        def run_analysis(self):
            if self.data is None or not self.trials:
                self.say('Load Suite2p and behaviour first.', 'orange'); return
            sig, used = self.signal()
            self.say('Running on %s ...' % used)
            try:
                self.seq_results = sa.analyse(
                    sig, self.trials, self.fs_spin.value(),
                    pre_s=self.pre_spin.value(), post_s=self.post_spin.value(),
                    lag_s=self.lag_spin.value(), tail_s=self.tail_spin.value(),
                    baseline_s=self.base_spin.value(), alpha=self.alpha_spin.value())
            except Exception as exc:
                import traceback; traceback.print_exc()
                QMessageBox.critical(self, 'Analysis failed', str(exc)); return
            if not self.seq_results:
                self.say('No complete trials in the window.', 'orange'); return
            self.header = dict(params=dict(signal_used=used,
                                           fs=self.fs_spin.value(),
                                           lag_s=self.lag_spin.value(),
                                           tail_s=self.tail_spin.value(),
                                           pre_s=self.pre_spin.value(),
                                           post_s=self.post_spin.value(),
                                           baseline_s=self.base_spin.value(),
                                           alpha=self.alpha_spin.value()))
            self._fill_boxes()
            n = sum(int(r['stats']['responsive'].sum())
                    for r in self.seq_results.values())
            timing = '; '.join(
                '%s: %d x %.2f s (%s)'
                % (c, len(r['stats']['names']), r['stats'].get('seg_dur', 0.5),
                   r['stats'].get('seg_source', '?'))
                for c, r in sorted(self.seq_results.items()))
            self.say('Done. %d cue-responsive cells, signal=%s. Segments %s'
                     % (n, used, timing), 'green')
            self.set_cell(self.current_cell)

        def _fill_boxes(self):
            self.cue_box.clear(); self.seg_box.clear()
            cues = sorted(self.seq_results.keys())
            self.cue_box.addItems(cues)
            names = self.seq_results[cues[0]]['stats']['names']
            self.seg_box.addItems(['whole sequence'] + list(names))
            self.plot_cue_box.blockSignals(True)
            self.plot_cue_box.clear()
            self.plot_cue_box.addItems(['all cues'] + cues)
            self.plot_cue_box.blockSignals(False)

        def load_cached(self):
            path, _ = QFileDialog.getOpenFileName(self, 'Cached results', '',
                                                  'NumPy (*.npz)')
            if not path:
                return
            try:
                self.seq_results, self.header = sa.load_results(path)
            except Exception as exc:
                QMessageBox.critical(self, 'Load failed', str(exc)); return
            self._fill_boxes()
            p = self.header.get('params', {})
            self.say('Loaded cache. Parameters used: %s'
                     % ', '.join('%s=%s' % kv for kv in sorted(p.items())), 'blue')
            self.set_cell(0)

        def save_npz(self):
            if not self.seq_results:
                return
            if 'resp' not in list(self.seq_results.values())[0]:
                self.say('Cached results cannot be re-saved; re-run first.', 'orange')
                return
            path, _ = QFileDialog.getSaveFileName(
                self, 'Save results',
                os.path.join(self.data['folder'], 'seq_results.npz'), 'NumPy (*.npz)')
            if not path:
                return
            sa.save_results(path, self.seq_results, self.trials,
                            self.header['params'],
                            dict(suite2p=self.data['folder'],
                                 behavior=getattr(self, 'beh_path', '')))
            self.say('Saved %s' % path, 'green')

        def export_csv(self):
            if not self.seq_results:
                return
            import csv
            path, _ = QFileDialog.getSaveFileName(self, 'Export CSV',
                                                  'seq_cells.csv', 'CSV (*.csv)')
            if not path:
                return
            rows = sa.results_to_rows(self.seq_results, self.data['s2p_ids'])
            with open(path, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            self.say('Exported %d cells to %s' % (len(rows), path), 'green')

        def draw_map(self):
            if not self.seq_results:
                self.say('Run the analysis first.', 'orange'); return
            cue = self.cue_box.currentText()
            seg = self.seg_box.currentIndex() - 1
            seg = None if seg < 0 else seg
            kind = {0: 'amp', 1: 'pref', 2: 'latency'}[self.kind_box.currentIndex()]
            try:
                rm.add_response_maps(self.viewer, self.data['stat'], self.data['ops'],
                                     {cue: self.seq_results[cue]},
                                     self.data['cell_indices'], cues=[cue],
                                     kinds=(kind,), segment=seg,
                                     mask_unresponsive=self.mask_chk.isChecked())
                for lyr in self.viewer.layers:
                    if cue in lyr.name:
                        lyr.visible = True
                self.say('Drew %s map for %s.' % (kind, cue), 'green')
            except Exception as exc:
                import traceback; traceback.print_exc()
                self.say('Map failed: %s' % exc, 'red')

        # ---------------- cells ----------------
        def _n_cells(self):
            if not self.seq_results:
                return 0
            return len(list(self.seq_results.values())[0]['stats']['responsive'])

        def _any_responsive(self):
            masks = [r['stats']['responsive'] for r in self.seq_results.values()]
            return np.any(np.vstack(masks), axis=0)

        def step(self, d):
            n = self._n_cells()
            if not n:
                return
            i = self.current_cell
            if self.resp_only_chk.isChecked():
                ok = np.where(self._any_responsive())[0]
                if ok.size:
                    pos = int(np.searchsorted(ok, i))
                    pos = (pos + d) % ok.size
                    i = int(ok[pos])
                else:
                    i = (i + d) % n
            else:
                i = (i + d) % n
            self.set_cell(i)

        def goto_id(self):
            if self.data is None:
                return
            want = self.id_spin.value()
            hit = np.where(self.data['s2p_ids'] == want)[0]
            if hit.size:
                self.set_cell(int(hit[0]))
            else:
                self.say('ROI %d is not a classified cell.' % want, 'orange')

        def set_cell(self, i):
            n = self._n_cells()
            if not n:
                return
            self.current_cell = int(np.clip(i, 0, n - 1))
            sid = int(self.data['s2p_ids'][self.current_cell]) if self.data else -1
            bits = []
            for cue, r in sorted(self.seq_results.items()):
                st = r['stats']
                bits.append('%s: %s%s' % (cue, st['pref_name'][self.current_cell],
                                          '' if st['responsive'][self.current_cell] else ' (n.s.)'))
            self.cell_lbl.setText('Cell %d/%d - s2p ROI %d - %s'
                                  % (self.current_cell + 1, n, sid, ' | '.join(bits)))
            if self.data is not None:
                self.id_spin.blockSignals(True)
                self.id_spin.setValue(sid)
                self.id_spin.blockSignals(False)
                try:
                    self.labels_layer.selected_label = self.current_cell + 1
                except Exception:
                    pass
            self.refresh_plot()

        def refresh_plot(self):
            if not self.seq_results:
                return
            kind = PLOTS[self.plot_box.currentText()]
            fig = self.canvas.fig
            try:
                if kind == 'psth':
                    plot_cell_psth(fig, self.seq_results, self.current_cell)
                elif kind == 'tuning':
                    plot_segment_tuning(fig, self.seq_results, self.current_cell)
                elif kind == 'population':
                    cue = self.plot_cue_box.currentText() or 'all cues'
                    sort_by = sorted(self.seq_results)[0] \
                        if self.lock_order_chk.isChecked() else None
                    plot_population_sequence(fig, self.seq_results, cue,
                                             self.mask_chk.isChecked(),
                                             sort_by=sort_by)
                elif kind == 'latency':
                    plot_latency_histogram(fig, self.seq_results)
                self.canvas.canvas.draw_idle()
            except Exception as exc:
                import traceback; traceback.print_exc()
                self.say('Plot failed: %s' % exc, 'red')

        def save_fig(self):
            path, _ = QFileDialog.getSaveFileName(self, 'Save figure', 'figure.png',
                                                  'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
            if path:
                self.canvas.fig.savefig(path, dpi=200, bbox_inches='tight')
                self.say('Saved %s' % path, 'green')

    return napari, SequenceGUI


def main():
    napari, SequenceGUI = build_gui()
    viewer = napari.Viewer(title='Sequence analysis')
    gui = SequenceGUI(viewer)
    viewer.window.add_dock_widget(gui, area='right', name='Sequence analysis')
    napari.run()


if __name__ == '__main__':
    main()
