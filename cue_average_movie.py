#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cue_average_movie.py

Trial-averaged dF/F movies, one per cue, from 3 s before cue onset to 3 s
after cue offset, with an on-screen indicator of when the stimulus was up.

This reads Suite2p's REGISTERED binary (data.bin), not the raw tiffs, so the
frames are already motion-corrected and frame-aligned with F.npy.

    python cue_average_movie.py --suite2p /path/to/suite2p/plane0 \
                                --behavior /path/to/behavior.txt

Outputs into --out (default ./cue_movies/):
    mean_<cue>.tif        float32 dF/F stack, opens in ImageJ or napari
    mean_<cue>.mp4        rendered with a stimulus indicator bar
    mean_<cue>_meta.json  timing, trial count, window definition

Add --napari to open the stacks in a viewer instead of / as well as writing.

Memory: one accumulator of (n_window, Ly, Lx) float32 per cue is held at a
time. For 512x512 and a 7.5 s window at 30.9 Hz that is ~240 MB. Use
--bin 2 to halve the linear resolution if that is tight.
"""

from __future__ import division
import argparse
import json
import os
import sys
import numpy as np

# --- make sure the sibling modules are importable, wherever this is run from ---
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import sequence_analysis as sa
except ImportError as _exc:
    _need = ['sequence_analysis.py']
    _missing = [f for f in _need if not os.path.isfile(os.path.join(_HERE, f))]
    raise SystemExit(
        "\ncue_average_movie.py could not import its companion modules.\n"
        "  error   : %s\n"
        "  folder  : %s\n"
        "  missing : %s\n\n"
        "Put %s in the SAME folder as this script, then run it again.\n"
        % (_exc, _HERE, ', '.join(_missing) if _missing else 'none (import failed for another reason)',
           ' and '.join(_need)))


# ---------------------------------------------------------------------
def open_registered(ops, suite2p_dir):
    """Memory-map Suite2p's registered binary. Returns (movie, Ly, Lx, n)."""
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    candidates = []
    for key in ('reg_file', 'raw_file'):
        if ops.get(key):
            candidates.append(str(ops[key]))
    candidates += [os.path.join(suite2p_dir, n)
                   for n in ('data.bin', 'data_raw.bin')]

    for path in candidates:
        if path and os.path.isfile(path):
            n = os.path.getsize(path) // (2 * Ly * Lx)
            mov = np.memmap(path, dtype='int16', mode='r', shape=(n, Ly, Lx))
            print('registered binary: %s  (%d frames, %dx%d)' % (path, n, Ly, Lx))
            return mov, Ly, Lx, n
    raise SystemExit(
        "\nNo registered binary (data.bin) found.\n"
        "Suite2p deletes it unless you ran with delete_bin=False.\n"
        "Without it there are no pixels to average - the movies need the\n"
        "registered movie, not just F.npy.\n"
        "Looked in:\n  %s\n" % '\n  '.join(str(c) for c in candidates))


def spatial_bin(a, b):
    """Bin the last two axes by integer factor b."""
    if b <= 1:
        return a
    n, y, x = a.shape
    y2, x2 = (y // b) * b, (x // b) * b
    return a[:, :y2, :x2].reshape(n, y2 // b, b, x2 // b, b).mean(axis=(2, 4))


def cue_average(mov, trials, cue, fs, pre_s=3.0, post_s=3.0, stim_s=1.5,
                bin_factor=1, f0_s=None, max_trials=None, verbose=True):
    """Trial-averaged dF/F movie for one cue.

    Window runs from -pre_s to stim_s + post_s relative to cue onset.
    F0 is the per-pixel mean over the f0_s seconds before onset
    (default: the whole pre-stimulus period).

    Returns (mean_dff, tvec, n_used).
    """
    sel = [t for t in trials if t['cue'] == cue]
    if max_trials:
        sel = sel[:max_trials]
    pre = int(round(pre_s * fs))
    post = int(round((stim_s + post_s) * fs))
    win = pre + post
    f0n = pre if f0_s is None else min(int(round(f0_s * fs)), pre)

    n_frames = mov.shape[0]
    acc, used = None, 0
    for tr in sel:
        a = tr['onset_frame'] - pre
        b = a + win
        if a < 0 or b > n_frames:
            continue
        blk = np.asarray(mov[a:b], dtype=np.float32)
        blk = spatial_bin(blk, bin_factor)
        f0 = blk[pre - f0n:pre].mean(axis=0)
        f0 = np.where(np.abs(f0) < 1e-3, np.nan, f0)      # avoid /0
        dff = (blk - f0) / f0
        acc = dff if acc is None else acc + dff
        used += 1
        if verbose and used % 10 == 0:
            print('    %s: %d/%d trials' % (cue, used, len(sel)))

    if used == 0:
        return None, None, 0
    tvec = (np.arange(win) - pre) / fs
    return acc / used, tvec, used


# ---------------------------------------------------------------------
def add_indicator_strip(stack, tvec, stim_s, seg_onsets, seg_dur,
                        height_frac=0.09, min_h=14):
    """Append a timeline strip BELOW the image showing when the stimulus ran.

    Extra rows are added rather than drawing over the field of view, so no
    real pixel is overwritten - the top Ly rows of every frame are still the
    untouched dF/F image.

    The strip shows: a dim track (whole window), a brighter band for the
    stimulus period, bright ticks at each segment boundary, and a moving
    cursor at the current frame.
    """
    n, Ly, Lx = stack.shape
    h = max(int(round(Ly * height_frac)), min_h)
    hi = float(np.nanpercentile(np.abs(stack), 99.5))
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0

    t0, t1 = float(tvec[0]), float(tvec[-1])
    span = (t1 - t0) if t1 > t0 else 1.0

    def x_of(t):
        return int(np.clip((t - t0) / span * (Lx - 1), 0, Lx - 1))

    track = np.zeros((h, Lx), np.float32)
    track[2:h - 2, :] = hi * 0.12                       # whole window
    track[2:h - 2, x_of(0.0):x_of(stim_s) + 1] = hi * 0.45   # stimulus period
    for s0 in list(seg_onsets) + [stim_s]:              # boundaries
        x = x_of(float(s0))
        track[1:h - 1, x:x + 1] = hi * 0.95

    strip = np.repeat(track[None], n, axis=0)
    for i in range(n):
        xc = x_of(float(tvec[i]))
        strip[i, :, max(xc - 1, 0):xc + 2] = hi        # cursor
    return np.concatenate([stack, strip], axis=1)


def imagej_labels(tvec, stim_s, seg_onsets, seg_names, seg_dur):
    """Per-slice labels so ImageJ's slice bar reads e.g. '+0.62 s STIM right'."""
    labs = []
    for t in tvec:
        tag = ''
        if 0 <= t < stim_s:
            k = int(np.searchsorted(np.asarray(seg_onsets, float), t,
                                    side='right') - 1)
            k = int(np.clip(k, 0, max(len(seg_names) - 1, 0)))
            nm = seg_names[k] if seg_names else 'seg%d' % (k + 1)
            tag = '  STIM %s' % nm
        labs.append('%+.2f s%s' % (t, tag))
    return labs


# ---------------------------------------------------------------------
def write_mp4(path, stack, tvec, cue, stim_s, seg_names=None, seg_dur=0.5,
              fps=20, percentile=99.0, cmap='inferno'):
    """Render with a stimulus indicator: a timeline bar below the image that
    fills during the stimulus, plus per-segment ticks and a moving cursor."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    vmax = np.nanpercentile(np.abs(stack), percentile)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0

    fig = plt.figure(figsize=(6.0, 6.8), facecolor='black')
    ax = fig.add_axes([0.02, 0.16, 0.96, 0.80])
    bar = fig.add_axes([0.10, 0.06, 0.80, 0.06])

    im = ax.imshow(np.nan_to_num(stack[0]), cmap=cmap, vmin=-vmax * 0.3,
                   vmax=vmax, interpolation='nearest')
    ax.axis('off')
    title = ax.set_title('%s   t = %+.2f s' % (cue, tvec[0]),
                         color='w', fontsize=11)

    bar.set_xlim(tvec[0], tvec[-1]); bar.set_ylim(0, 1)
    bar.set_facecolor('black')
    bar.add_patch(Rectangle((0, 0), stim_s, 1, facecolor='#3aa0e8', alpha=0.55))
    if seg_names:
        for k, nm in enumerate(seg_names):
            bar.axvline(k * seg_dur, color='w', lw=0.8, alpha=0.8)
            bar.text(k * seg_dur + seg_dur / 2, 0.5, nm, ha='center',
                     va='center', color='w', fontsize=7)
    cursor = bar.axvline(tvec[0], color='#ff4444', lw=2)
    bar.set_yticks([]); bar.tick_params(colors='w', labelsize=8)
    bar.set_xlabel('time from cue onset (s)', color='w', fontsize=9)
    for sp in bar.spines.values():
        sp.set_color('0.4')

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    frames = []
    for i in range(stack.shape[0]):
        im.set_data(np.nan_to_num(stack[i]))
        title.set_text('%s   t = %+.2f s%s'
                       % (cue, tvec[i], '   [STIM]' if 0 <= tvec[i] < stim_s else ''))
        cursor.set_xdata([tvec[i], tvec[i]])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(buf.copy())
    plt.close(fig)

    try:
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        return path
    except Exception as exc:
        gif = path.replace('.mp4', '.gif')
        print('    mp4 failed (%s), writing %s instead' % (exc, gif))
        imageio.mimwrite(gif, frames, duration=1.0 / fps, loop=0)
        return gif


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite2p', required=True, help='.../suite2p/plane0')
    ap.add_argument('--behavior', required=True)
    ap.add_argument('--out', default='cue_movies')
    ap.add_argument('--fs', type=float, default=None, help='default: ops["fs"]')
    ap.add_argument('--pre', type=float, default=3.0)
    ap.add_argument('--post', type=float, default=3.0, help='after cue OFFSET')
    ap.add_argument('--stim', type=float, default=None,
                    help='cue duration; default: median measured')
    ap.add_argument('--seg-dur', type=float, default=0.5)
    ap.add_argument('--bin', type=int, default=1, help='spatial binning factor')
    ap.add_argument('--max-trials', type=int, default=None)
    ap.add_argument('--fps', type=int, default=20, help='playback fps of the mp4')
    ap.add_argument('--no-mp4', action='store_true')
    ap.add_argument('--no-marked', action='store_true',
                    help='skip the extra mean_<cue>_marked.tif with the '
                         'burned-in stimulus indicator strip')
    ap.add_argument('--napari', action='store_true')
    args = ap.parse_args()

    s2p = args.suite2p
    if os.path.isdir(os.path.join(s2p, 'plane0')):
        s2p = os.path.join(s2p, 'plane0')       # accept suite2p/ too
    if not os.path.isfile(os.path.join(s2p, 'ops.npy')):
        raise SystemExit(
            "\nNo ops.npy in %s\n"
            "Point --suite2p at the folder containing ops.npy and data.bin,\n"
            "usually  ...\\suite2p\\plane0\n" % s2p)
    args.suite2p = s2p
    ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
    fs = args.fs or float(ops.get('fs', 30.9))
    mov, Ly, Lx, n_frames = open_registered(ops, args.suite2p)

    trials, _ = sa.load_stim_trials(args.behavior, n_frames_expected=n_frames)
    if not trials:
        raise SystemExit('No trials found in the behaviour file.')

    stim_s = args.stim or float(np.median([t['duration_s'] for t in trials]))
    os.makedirs(args.out, exist_ok=True)
    print('\nwindow: -%.1f to +%.1f s around a %.2f s stimulus  (bin %d)'
          % (args.pre, stim_s + args.post, stim_s, args.bin))

    try:
        import tifffile
    except ImportError:
        raise SystemExit(
            "\ntifffile is not installed in this environment.\n"
            "    conda install -c conda-forge tifffile imageio imageio-ffmpeg\n")
    if not args.no_mp4:
        try:
            import imageio  # noqa: F401
        except ImportError:
            print('\n*** imageio not installed - writing TIFFs only.')
            print('    conda install -c conda-forge imageio imageio-ffmpeg\n')
            args.no_mp4 = True

    stacks = {}
    for cue in sorted(set(t['cue'] for t in trials)):
        print('\n%s' % cue)
        stack, tvec, used = cue_average(
            mov, trials, cue, fs, args.pre, args.post, stim_s,
            args.bin, max_trials=args.max_trials)
        if stack is None:
            print('  no complete trials, skipped')
            continue
        print('  averaged %d trials -> %s' % (used, stack.shape))
        stacks[cue] = (stack, tvec)

        seg_names = next((t['seg_names'] for t in trials if t['cue'] == cue), None)
        seg_onsets = next((list(t['seg_onset_t']) for t in trials
                           if t['cue'] == cue), [0.0])
        seg_dur = next((float(t['seg_dur']) for t in trials
                        if t['cue'] == cue), args.seg_dur)

        # 1) untouched data, but with ImageJ slice labels and a frame interval
        #    so the slice bar shows the time and which segment is running
        tif = os.path.join(args.out, 'mean_%s.tif' % cue)
        try:
            tifffile.imwrite(
                tif, stack.astype(np.float32), imagej=True,
                metadata={'axes': 'TYX', 'unit': 's', 'finterval': 1.0 / fs,
                          'fps': float(fs),
                          'Labels': imagej_labels(tvec, stim_s, seg_onsets,
                                                  seg_names, seg_dur)})
        except Exception:
            tifffile.imwrite(tif, stack.astype(np.float32))
        print('  wrote %s' % tif)

        # 2) same movie with a burned-in indicator strip below the image,
        #    for napari or anything that ignores slice labels
        if not args.no_marked:
            marked = add_indicator_strip(stack, tvec, stim_s, seg_onsets, seg_dur)
            mtif = os.path.join(args.out, 'mean_%s_marked.tif' % cue)
            tifffile.imwrite(mtif, marked.astype(np.float32), imagej=True,
                             metadata={'axes': 'TYX', 'unit': 's',
                                       'finterval': 1.0 / fs})
            print('  wrote %s  (image rows 0-%d, indicator strip below)'
                  % (mtif, stack.shape[1] - 1))

        # 3) a plain per-frame table of what the stimulus was doing
        icsv = os.path.join(args.out, 'mean_%s_indicator.csv' % cue)
        with open(icsv, 'w') as fh:
            fh.write('frame,time_s,in_stimulus,segment_index,segment_name\n')
            for i, t in enumerate(tvec):
                inside = 1 if (0 <= t < stim_s) else 0
                k, nm = -1, ''
                if inside:
                    k = int(np.clip(np.searchsorted(np.asarray(seg_onsets, float),
                                                    t, side='right') - 1,
                                    0, max(len(seg_names or [1]) - 1, 0)))
                    nm = seg_names[k] if seg_names else ''
                fh.write('%d,%.4f,%d,%d,%s\n' % (i, t, inside, k, nm))
        print('  wrote %s' % icsv)
        meta = dict(cue=cue, n_trials=used, fs=fs, pre_s=args.pre,
                    post_s=args.post, stim_s=stim_s, bin=args.bin,
                    seg_names=seg_names, seg_dur=args.seg_dur,
                    onset_frame_in_stack=int(round(args.pre * fs)),
                    offset_frame_in_stack=int(round((args.pre + stim_s) * fs)),
                    image_rows=int(stack.shape[1]),
                    seg_onsets_s=[float(x) for x in seg_onsets],
                    tvec_first=float(tvec[0]), tvec_last=float(tvec[-1]),
                    shape=list(stack.shape))
        with open(os.path.join(args.out, 'mean_%s_meta.json' % cue), 'w') as fh:
            json.dump(meta, fh, indent=2)

        if not args.no_mp4:
            p = write_mp4(os.path.join(args.out, 'mean_%s.mp4' % cue),
                          stack, tvec, cue, stim_s, seg_names, seg_dur,
                          fps=args.fps)
            print('  wrote %s' % p)

    if args.napari and stacks:
        import napari
        v = napari.Viewer()
        for cue, (stack, tvec) in stacks.items():
            vmax = float(np.nanpercentile(np.abs(stack), 99))
            v.add_image(np.nan_to_num(stack), name='mean %s' % cue,
                        colormap='inferno', contrast_limits=(-0.3 * vmax, vmax))
        onset = int(round(args.pre * fs))
        print('\nIn napari the stimulus runs from frame %d to %d of the stack.'
              % (onset, onset + int(round(stim_s * fs))))
        napari.run()


if __name__ == '__main__':
    main()
