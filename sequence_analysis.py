# -*- coding: utf-8 -*-
"""
sequence_analysis.py

Segment-aware analysis for the 3-part noise sequence stimulus.

The problem this solves: the existing pipeline detects a rising edge, then asks
"did this cell respond in the first second?". With a sequence stimulus that is
wrong. A cell whose receptive field sits in the THIRD segment responds at
1.0-1.5 s after cue onset and is scored as unresponsive. Here every segment
gets its own response window, and a cell counts as responsive if ANY segment
drives it - which also tells you WHICH segment, i.e. its retinotopic position.

Nothing here imports napari or Qt, so it can be used from a script, a notebook,
or the GUI.

Main entry points
    load_stim_trials()        parallel-port channels -> trial table with
                              cue identity, onset/offset and segment boundaries
    build_trial_tensor()      (n_trials, n_cells, n_window) aligned to cue onset
    compute_segment_responses()  per-trial amplitude for every segment
    test_responsiveness()     per-segment stats, corrected, + preferred segment
    peak_latency()            when each cell peaks - the sequence readout

@author: analysis add-on for krisklen's napari GUI
"""

from __future__ import division
import re
import numpy as np

try:
    from scipy import stats as _stats
except ImportError:
    _stats = None


# ---------------------------------------------------------------------
# CONFIGURATION - match this to how the PCB wires the port into the DAQ
# ---------------------------------------------------------------------
# The stimulus sends ONE code per cue sequence, held high for the whole
# 1.5 s, so on the DAQ side each cue is a single long pulse. Map the
# behaviour-file column that carries each cue line to the cue name.
DEFAULT_CUE_MAP = {
    'plus':  'cue1_azimuth',      # port line 0
    'minus': 'cue2_elevation',    # port line 1
}

# Optional third line pulsed at each segment onset. Set to None if unused.
DEFAULT_TICK_CHANNEL = 'neutral'

# Fixed segment order within each cue (must match `cues` in the stim script).
DEFAULT_SEGMENTS = {
    'cue1_azimuth':   ['left', 'centre_v', 'right'],
    'cue2_elevation': ['top',  'centre_h', 'bottom'],
}

# Nominal timing of the stimulus, used when ticks are absent or unreliable.
DEFAULT_SEGMENT_DUR = None    # None = infer from the recording
DEFAULT_N_SEGMENTS = 3

BEH_COLUMNS_10 = ['time', 'frame', 'running_fwd', 'running_bckw', 'licking',
                  'reward', 'plus', 'minus', 'neutral', 'aversive']
BEH_COLUMNS_9 = ['time', 'frame', 'running_fwd', 'licking',
                 'reward', 'plus', 'minus', 'neutral', 'aversive']


# ---------------------------------------------------------------------
# Behaviour file -> raw channel traces
# ---------------------------------------------------------------------
def read_behavior_raw(path):
    """Parse the behaviour text file WITHOUT collapsing to frames.

    The existing loader assigns one value per imaging frame (last sample
    wins), which is fine for a 1.5 s cue pulse but can silently drop a
    30 ms segment tick. Edges are therefore detected on the raw samples
    here, and only then mapped to frames.

    Returns dict of 1-D arrays keyed by column name, plus 'n_samples'.
    """
    def _f(tok):
        return float(tok.replace(',', '.'))

    def _frame(tok):
        tok = tok.replace(',', '.')
        m = re.match(r'^(-?\d+)', tok)
        return float(m.group(1)) if m else float(tok)

    content = None
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1', 'iso-8859-1'):
        try:
            with open(path, 'r', encoding=enc) as fh:
                content = fh.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if content is None:
        raise ValueError('Could not decode %s' % path)

    lines = content.splitlines()
    sample = next((l.strip() for l in lines if l.strip()), None)
    if sample is None:
        raise ValueError('Behaviour file is empty')
    delim = '\t' if '\t' in sample else (';' if ';' in sample else None)

    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        toks = line.split(delim) if delim else line.split()
        try:
            _f(toks[0])
        except (ValueError, IndexError):
            continue        # header
        rows.append(toks)
    if not rows:
        raise ValueError('No data rows in %s' % path)

    ncol = len(rows[0])
    if ncol >= 10:
        names = BEH_COLUMNS_10
    elif ncol == 9:
        names = BEH_COLUMNS_9
    else:
        raise ValueError('Unexpected column count %d (expected 9 or 10)' % ncol)

    out = {nm: np.zeros(len(rows)) for nm in names}
    for i, row in enumerate(rows):
        for j, nm in enumerate(names):
            if j < len(row):
                out[nm] = out[nm]
                out[nm][i] = _frame(row[j]) if nm == 'frame' else _f(row[j])
    out['frame'] = out['frame'].astype(int)
    out['n_samples'] = len(rows)
    return out


def _edges(signal, thresh):
    """Rising and falling sample indices of a square pulse train."""
    above = (signal > thresh).astype(int)
    d = np.diff(above)
    rise = np.where(d == 1)[0] + 1
    fall = np.where(d == -1)[0] + 1
    if above[0] == 1:                      # already high at file start
        rise = np.concatenate([[0], rise])
    if above[-1] == 1:                     # still high at file end
        fall = np.concatenate([fall, [len(signal)]])
    n = min(len(rise), len(fall))
    return rise[:n], fall[:n]


# ---------------------------------------------------------------------
# Trial table
# ---------------------------------------------------------------------
def load_stim_trials(path, cue_map=None, tick_channel=DEFAULT_TICK_CHANNEL,
                     segments=None, n_frames_expected=None, threshold=2.0,
                     segment_dur=DEFAULT_SEGMENT_DUR,
                     n_segments=DEFAULT_N_SEGMENTS, verbose=True):
    """Build a trial table from the cue lines in the behaviour file.

    Each trial is a dict:
        cue            'cue1_azimuth' / 'cue2_elevation'
        channel        behaviour-file column it came from
        onset_frame    imaging frame of the rising edge
        offset_frame   imaging frame of the falling edge
        onset_t        seconds, from the file's time column
        duration_s     measured cue duration
        seg_names      ['left', 'centre_v', 'right']
        seg_onset_t    segment onset times relative to cue onset (s)
        seg_source     'ticks' or 'nominal'

    Segment boundaries come from the tick line when it is present and the
    right number of ticks was seen; otherwise the cue is divided into
    n_segments equal parts of segment_dur.
    """
    cue_map = DEFAULT_CUE_MAP if cue_map is None else cue_map
    segments = DEFAULT_SEGMENTS if segments is None else segments

    raw = read_behavior_raw(path)
    t = raw['time']
    fr = raw['frame']
    max_frame = int(fr.max())
    n_frames = max_frame + 1
    if n_frames_expected is not None:
        n_frames = min(n_frames, int(n_frames_expected))

    # tick edges (raw sample resolution)
    tick_t = np.array([])
    if tick_channel and tick_channel in raw:
        tr, _ = _edges(raw[tick_channel], threshold)
        tick_t = t[np.clip(tr, 0, len(t) - 1)]

    trials = []
    for chan, cue in cue_map.items():
        if chan not in raw:
            if verbose:
                print('  channel %r not in behaviour file, skipping' % chan)
            continue
        rise, fall = _edges(raw[chan], threshold)
        for r, f in zip(rise, fall):
            r = int(np.clip(r, 0, len(t) - 1))
            f = int(np.clip(f, 0, len(t) - 1))
            onset_frame = int(fr[r])
            offset_frame = int(fr[f])
            if onset_frame >= n_frames:
                continue
            dur = float(t[f] - t[r])

            seg_names = list(segments.get(cue, []))
            inside = tick_t[(tick_t >= t[r] - 1e-6) & (tick_t < t[f] - 1e-6)]

            if len(inside) >= 2:
                # ticks give the real boundaries, whatever you set epoch_dur to
                nseg = len(inside)
                seg_rel = (inside - t[r]).astype(float)
                seg_durs = np.diff(np.append(seg_rel, dur))
                seg_source = 'ticks'
            elif segment_dur is None:
                # no ticks: divide the MEASURED cue duration, so a changed
                # epoch_dur is still handled correctly
                nseg = len(seg_names) if seg_names else int(n_segments)
                d = dur / float(nseg)
                seg_rel = np.arange(nseg) * d
                seg_durs = np.full(nseg, d, float)
                seg_source = 'divided'
            else:
                nseg = len(seg_names) if seg_names else int(n_segments)
                seg_rel = np.arange(nseg) * float(segment_dur)
                seg_durs = np.full(nseg, float(segment_dur), float)
                seg_source = 'nominal'

            if len(seg_names) != nseg:
                seg_names = (seg_names +
                             ['seg%d' % k for k in range(len(seg_names), nseg)])[:nseg]

            trials.append(dict(
                cue=cue, channel=chan,
                onset_frame=onset_frame, offset_frame=offset_frame,
                onset_t=float(t[r]), offset_t=float(t[f]),
                duration_s=dur,
                seg_names=list(seg_names),
                seg_onset_t=np.asarray(seg_rel, dtype=float),
                seg_durs=np.asarray(seg_durs, dtype=float),
                seg_dur=float(np.median(seg_durs)),
                n_segments=int(nseg),
                seg_source=seg_source,
            ))

    trials.sort(key=lambda d: d['onset_frame'])
    for i, tr_ in enumerate(trials):
        tr_['trial'] = i

    if verbose:
        print('loaded %d trials from %s' % (len(trials), path))
        for cue in sorted(set(d['cue'] for d in trials)):
            sub = [d for d in trials if d['cue'] == cue]
            durs = np.array([d['duration_s'] for d in sub])
            ntick = sum(1 for d in sub if d['seg_source'] == 'ticks')
            nsegs = sorted(set(d['n_segments'] for d in sub))
            segd = np.array([d['seg_dur'] for d in sub])
            print('  %-16s n=%3d  cue %.3f +/- %.3f s  |  %s segment(s) of '
                  '%.3f +/- %.3f s  |  source: %s'
                  % (cue, len(sub), durs.mean(), durs.std(),
                     '/'.join(str(x) for x in nsegs), segd.mean(), segd.std(),
                     ', '.join(sorted(set(d['seg_source'] for d in sub)))))
            if len(nsegs) > 1:
                print('    *** trials disagree on the number of segments %s; '
                      'trials not matching the commonest count will be dropped'
                      % nsegs)
            if durs.std() > 0.05 * max(durs.mean(), 1e-9):
                print('    *** cue duration is not constant (sd %.3f s) - check '
                      'the port wiring' % durs.std())
    return trials, n_frames


# ---------------------------------------------------------------------
# Aligning the signal
# ---------------------------------------------------------------------
def build_trial_tensor(sig, trials, fs, pre_s, post_s, cue=None):
    """Cut sig (n_cells, n_frames) around each cue onset.

    post_s is measured from cue ONSET, so pass at least stimulus length +
    the decay you want to see. Returns (tensor, kept_trials, tvec) with
    tensor shaped (n_trials, n_cells, n_window).
    """
    sig = np.asarray(sig)
    n_cells, n_frames = sig.shape
    pre = int(round(pre_s * fs))
    post = int(round(post_s * fs))
    win = pre + post

    keep, cuts = [], []
    for tr in trials:
        if cue is not None and tr['cue'] != cue:
            continue
        a = tr['onset_frame'] - pre
        b = a + win
        if a < 0 or b > n_frames:
            continue
        cuts.append(sig[:, a:b])
        keep.append(tr)

    if not cuts:
        return np.zeros((0, n_cells, win)), [], (np.arange(win) - pre) / fs
    return np.asarray(cuts), keep, (np.arange(win) - pre) / fs


def segment_windows(trial, fs, pre_s, lag_s=0.10, tail_s=0.30):
    """Index windows (into a tensor built with the same pre_s) for each
    segment of one trial.

    lag_s shifts the window to allow for indicator rise time; tail_s
    extends it past segment offset so a response that starts late still
    lands inside. With 0.5 s segments and a slow indicator these windows
    OVERLAP - see the note in `compute_segment_responses`.
    """
    pre = int(round(pre_s * fs))
    durs = trial.get('seg_durs')
    if durs is None:
        durs = np.full(len(trial['seg_names']), trial['seg_dur'], float)
    out = []
    for name, rel, d in zip(trial['seg_names'], trial['seg_onset_t'], durs):
        a = pre + int(round((rel + lag_s) * fs))
        b = pre + int(round((rel + d + lag_s + tail_s) * fs))
        out.append((name, max(a, 0), max(b, a + 1)))
    return out


def compute_segment_responses(sig, trials, fs, pre_s=2.0, post_s=5.0,
                              lag_s=0.10, tail_s=0.30, baseline_s=1.0,
                              cue=None):
    """Per-trial, per-segment response amplitude.

    amplitude = mean(signal in segment window) - mean(signal in baseline)
    baseline  = the baseline_s seconds immediately before cue onset

    IMPORTANT: with 0.5 s segments the windows overlap in time once lag and
    tail are added, and a calcium indicator smears energy from one segment
    into the next regardless. Segment-resolved numbers from raw dF/F should
    be read as "which segment drives this cell MOST", not as independent
    measurements. For anything stronger, pass Suite2p's deconvolved `spks`
    instead of dF/F - see deconvolved_note().

    Returns dict with:
        amp     (n_trials, n_cells, n_segments)
        names   segment names
        tensor  (n_trials, n_cells, n_window) aligned to onset
        tvec    seconds relative to onset
        trials  the trials actually used
    """
    tensor, kept, tvec = build_trial_tensor(sig, trials, fs, pre_s, post_s, cue)
    if tensor.shape[0] == 0:
        return None

    # keep only trials with the commonest segment count, otherwise the
    # amplitude array has no consistent shape
    counts = [len(t['seg_names']) for t in kept]
    nseg = int(np.bincount(counts).argmax())
    good = [i for i, c in enumerate(counts) if c == nseg]
    if len(good) != len(kept):
        print('  %s: dropped %d trial(s) with a different segment count'
              % (cue, len(kept) - len(good)))
        tensor = tensor[good]
        kept = [kept[i] for i in good]

    pre = int(round(pre_s * fs))
    b0 = max(pre - int(round(baseline_s * fs)), 0)
    base = tensor[:, :, b0:pre].mean(axis=2)                 # (n_trials, n_cells)
    amp = np.zeros((tensor.shape[0], tensor.shape[1], nseg))
    for i, tr in enumerate(kept):
        for k, (_nm, a, b) in enumerate(segment_windows(tr, fs, pre_s, lag_s, tail_s)):
            b = min(b, tensor.shape[2])
            amp[i, :, k] = tensor[i, :, a:b].mean(axis=1) - base[i]

    return dict(amp=amp, names=list(kept[0]['seg_names']), tensor=tensor,
                tvec=tvec, trials=kept, pre=pre, baseline=base,
                seg_dur=float(np.median([t['seg_dur'] for t in kept])),
                seg_onsets=np.median([t['seg_onset_t'] for t in kept], axis=0),
                seg_source=kept[0]['seg_source'])


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------
def _holm(p):
    """Holm-Bonferroni correction along the last axis."""
    p = np.asarray(p, float)
    n = p.shape[-1]
    order = np.argsort(p, axis=-1)
    ranked = np.take_along_axis(p, order, axis=-1)
    factors = n - np.arange(n)
    adj = np.minimum.accumulate((ranked * factors)[..., ::-1], axis=-1)[..., ::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    np.put_along_axis(out, order, adj, axis=-1)
    return out


def fdr_bh(p, alpha=0.05):
    """Benjamini-Hochberg. Returns (reject, qvalues) for a flat array."""
    p = np.asarray(p, float).ravel()
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out <= alpha, out


def test_responsiveness(resp, alpha=0.05, min_trials=5, correct='holm'):
    """Per-cell, per-segment test that the amplitude differs from zero.

    Runs a two-sided t-test across trials for every (cell, segment), then
    corrects ACROSS SEGMENTS within a cell (Holm by default) so that
    "responsive to any segment" does not inflate the false positive rate
    just because there are three chances instead of one. A further BH
    correction across cells is applied to the per-cell minimum.

    Returns dict:
        p_seg        (n_cells, n_segments) raw p
        p_seg_corr   (n_cells, n_segments) corrected within cell
        p_cell       (n_cells,) per-cell minimum corrected p
        responsive   (n_cells,) bool after BH across cells
        mean_amp     (n_cells, n_segments) mean amplitude
        pref_index   (n_cells,) index of strongest segment
        pref_name    (n_cells,) name of strongest segment
        n_trials
    """
    amp = resp['amp']                       # (trials, cells, segs)
    n_trials, n_cells, n_seg = amp.shape
    mean_amp = amp.mean(axis=0)
    if n_trials < min_trials:
        raise ValueError('only %d trials, need >= %d' % (n_trials, min_trials))

    if _stats is not None:
        tstat, p_seg = _stats.ttest_1samp(amp, 0.0, axis=0)
        p_seg = np.nan_to_num(p_seg, nan=1.0)
    else:                                    # normal approximation fallback
        sd = amp.std(axis=0, ddof=1)
        sd[sd == 0] = np.inf
        z = mean_amp / (sd / np.sqrt(n_trials))
        p_seg = 2 * 0.5 * np.exp(-0.717 * np.abs(z) - 0.416 * z ** 2)
        p_seg = np.clip(p_seg, 0, 1)

    p_corr = _holm(p_seg) if correct == 'holm' else np.clip(p_seg * n_seg, 0, 1)
    p_cell = p_corr.min(axis=1)
    responsive, q_cell = fdr_bh(p_cell, alpha)

    pref = np.argmax(mean_amp, axis=1)
    names = np.asarray(resp['names'])
    return dict(p_seg=p_seg, p_seg_corr=p_corr, p_cell=p_cell, q_cell=q_cell,
                responsive=responsive, mean_amp=mean_amp,
                pref_index=pref, pref_name=names[pref],
                names=list(names), n_trials=n_trials)


def whole_stimulus_response(resp, alpha=0.05):
    """The simple alternative: one window spanning the WHOLE sequence.

    More power when a cell responds weakly to several segments, less
    informative about which position drove it. Worth reporting alongside
    the segment-wise test - if the two disagree a lot, that itself is
    informative about how distributed the drive is.
    """
    amp_all = resp['amp'].mean(axis=2)       # (trials, cells)
    if _stats is not None:
        _, p = _stats.ttest_1samp(amp_all, 0.0, axis=0)
        p = np.nan_to_num(p, nan=1.0)
    else:
        sd = amp_all.std(axis=0, ddof=1); sd[sd == 0] = np.inf
        z = amp_all.mean(axis=0) / (sd / np.sqrt(amp_all.shape[0]))
        p = np.clip(2 * 0.5 * np.exp(-0.717 * np.abs(z) - 0.416 * z ** 2), 0, 1)
    responsive, q = fdr_bh(p, alpha)
    return dict(p=p, q=q, responsive=responsive, mean_amp=amp_all.mean(axis=0))


# ---------------------------------------------------------------------
# Sequence readout
# ---------------------------------------------------------------------
def peak_latency(resp, smooth_s=0.15, fs=None, search=(0.0, None)):
    """Time of each cell's peak in the trial-averaged trace, relative to
    cue onset. Sorting cells by this is the most direct way to see whether
    a travelling wave of activity actually happened.

    Returns dict: latency_s (n_cells,), peak (n_cells,), avg (n_cells, n_win)
    """
    tensor, tvec = resp['tensor'], resp['tvec']
    avg = tensor.mean(axis=0)                                  # (cells, win)

    if fs is not None and smooth_s and smooth_s > 0:
        w = max(int(round(smooth_s * fs)), 1)
        if w > 1:
            k = np.ones(w) / w
            avg = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 1, avg)

    lo, hi = search
    m = tvec >= (lo if lo is not None else tvec[0])
    if hi is not None:
        m &= tvec <= hi
    idx = np.argmax(avg[:, m], axis=1)
    return dict(latency_s=tvec[m][idx], peak=avg[:, m].max(axis=1), avg=avg,
                tvec=tvec)


def deconvolved_note():
    return (
        "Segment windows are 0.5 s but GCaMP decays over ~0.4 s (6f) to ~1.5 s "
        "(6s), so consecutive segments bleed into each other in dF/F. For "
        "segment-resolved claims load Suite2p's spks.npy instead of dF/F and "
        "pass it as `sig`; everything here works unchanged. Keep dF/F for the "
        "PSTH plots and the movies, where the smearing is cosmetic."
    )


# ---------------------------------------------------------------------
# Convenience: run the whole thing
# ---------------------------------------------------------------------
def analyse(sig, trials, fs, cues=None, pre_s=2.0, post_s=5.0,
            lag_s=0.10, tail_s=0.30, baseline_s=1.0, alpha=0.05, verbose=True):
    """Full per-cue analysis. Returns {cue: {...}}."""
    cues = cues or sorted(set(t['cue'] for t in trials))
    out = {}
    for cue in cues:
        resp = compute_segment_responses(sig, trials, fs, pre_s, post_s,
                                         lag_s, tail_s, baseline_s, cue=cue)
        if resp is None:
            if verbose:
                print('%s: no complete trials' % cue)
            continue
        st = test_responsiveness(resp, alpha=alpha)
        st['seg_dur'] = resp['seg_dur']
        st['seg_onsets'] = resp['seg_onsets']
        st['seg_source'] = resp['seg_source']
        wh = whole_stimulus_response(resp, alpha=alpha)
        lat = peak_latency(resp, fs=fs, search=(0.0, None))
        out[cue] = dict(resp=resp, stats=st, whole=wh, latency=lat)

        if verbose:
            n = len(st['responsive'])
            print('%s  (%d trials, %d cells, %d x %.3f s segments from %s)'
                  % (cue, st['n_trials'], n, len(st['names']), st['seg_dur'],
                     st['seg_source']))
            print('   responsive, segment-wise : %4d (%.1f%%)'
                  % (st['responsive'].sum(), 100.0 * st['responsive'].mean()))
            print('   responsive, whole window : %4d (%.1f%%)'
                  % (wh['responsive'].sum(), 100.0 * wh['responsive'].mean()))
            first_only = (st['pref_index'][st['responsive']] == 0).sum()
            print('   of the responsive cells, %d prefer segment 1; '
                  'preferred-segment counts %s'
                  % (first_only,
                     {nm: int((st['pref_name'][st['responsive']] == nm).sum())
                      for nm in st['names']}))
    return out


# ---------------------------------------------------------------------
# Persistence - so batch runs and the GUI share one set of numbers
# ---------------------------------------------------------------------
def save_results(path, results, trials, params, meta=None):
    """Write analysis output to a single .npz.

    The point of this is provenance. Every number the GUI displays should
    have come from a file that also records the parameters that produced
    it, so six months later you can tell what `tail_s` was.
    """
    import json as _json
    out, index = {}, {}
    for cue, r in results.items():
        st, lat = r['stats'], r['latency']
        for k in ('mean_amp', 'p_seg', 'p_seg_corr', 'p_cell', 'q_cell',
                  'responsive', 'pref_index'):
            out['%s|%s' % (cue, k)] = np.asarray(st[k])
        out['%s|pref_name' % cue] = np.asarray(st['pref_name']).astype('U32')
        out['%s|latency_s' % cue] = np.asarray(lat['latency_s'])
        out['%s|peak' % cue] = np.asarray(lat['peak'])
        out['%s|avg' % cue] = np.asarray(lat['avg'], dtype=np.float32)
        _tn = r['resp']['tensor']
        out['%s|sem' % cue] = (_tn.std(axis=0) /
                               max(np.sqrt(_tn.shape[0]), 1)).astype(np.float32)
        out['%s|tvec' % cue] = np.asarray(lat['tvec'])
        out['%s|whole_responsive' % cue] = np.asarray(r['whole']['responsive'])
        out['%s|whole_p' % cue] = np.asarray(r['whole']['p'])
        index[cue] = dict(names=list(st['names']), n_trials=int(st['n_trials']),
                          seg_dur=float(st.get('seg_dur', 0.5)),
                          seg_onsets=[float(x) for x in
                                      st.get('seg_onsets', [])],
                          seg_source=str(st.get('seg_source', 'unknown')))

    hdr = dict(params=dict(params), cues=index, meta=dict(meta or {}),
               n_trials_total=len(trials),
               trials=[dict(cue=t['cue'], onset_frame=int(t['onset_frame']),
                            offset_frame=int(t['offset_frame']),
                            duration_s=float(t['duration_s']),
                            seg_source=t['seg_source']) for t in trials])
    out['__header__'] = np.frombuffer(_json.dumps(hdr).encode('utf-8'), dtype=np.uint8)
    np.savez_compressed(path, **out)
    return path


def load_results(path):
    """Read back save_results(). Returns (results, header)."""
    import json as _json
    z = np.load(path, allow_pickle=False)
    hdr = _json.loads(bytes(z['__header__']).decode('utf-8'))
    results = {}
    for cue, info in hdr['cues'].items():
        g = lambda k: z['%s|%s' % (cue, k)]
        results[cue] = dict(
            stats=dict(mean_amp=g('mean_amp'), p_seg=g('p_seg'),
                       p_seg_corr=g('p_seg_corr'), p_cell=g('p_cell'),
                       q_cell=g('q_cell'), responsive=g('responsive'),
                       pref_index=g('pref_index'), pref_name=g('pref_name'),
                       names=info['names'], n_trials=info['n_trials'],
                       seg_dur=info.get('seg_dur', 0.5),
                       seg_onsets=np.asarray(info.get('seg_onsets', [])),
                       seg_source=info.get('seg_source', 'unknown')),
            latency=dict(latency_s=g('latency_s'), peak=g('peak'),
                         avg=g('avg'), tvec=g('tvec'), sem=g('sem')),
            whole=dict(responsive=g('whole_responsive'), p=g('whole_p')),
        )
    return results, hdr


def results_to_rows(results, s2p_ids=None):
    """Flatten to one dict per cell, ready for csv.DictWriter."""
    cues = sorted(results.keys())
    n = len(results[cues[0]]['stats']['responsive'])
    ids = np.arange(1, n + 1) if s2p_ids is None else np.asarray(s2p_ids)
    rows = []
    for i in range(n):
        row = {'s2p_roi_id': int(ids[i])}
        for cue in cues:
            st, lat = results[cue]['stats'], results[cue]['latency']
            row['%s_responsive' % cue] = bool(st['responsive'][i])
            row['%s_whole_responsive' % cue] = bool(results[cue]['whole']['responsive'][i])
            row['%s_pref_segment' % cue] = str(st['pref_name'][i])
            row['%s_peak_latency_s' % cue] = float(lat['latency_s'][i])
            row['%s_p_corrected' % cue] = float(st['p_cell'][i])
            for k, nm in enumerate(st['names']):
                row['%s_%s_amp' % (cue, nm)] = float(st['mean_amp'][i, k])
                row['%s_%s_p' % (cue, nm)] = float(st['p_seg_corr'][i, k])
        rows.append(row)
    return rows
