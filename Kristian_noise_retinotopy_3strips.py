#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Binarized bandpass-filtered noise, presented in screen thirds.

Built from the Kristian_functional_190909.py PsychoPy3 Builder template
(v3.1.5) - same window setup, same loop scaffolding, same parallel-port
component format.

Instead of drifting gratings this presents sequences of noise:
Each cue is a three-step sequence of 0.5 s noise presentations, designed to
drive a travelling wave of activity across retinotopic cortex:

    Cue 1 (azimuth)   : LEFT -> CENTRE -> RIGHT
    Cue 2 (elevation) : TOP  -> CENTRE -> BOTTOM

Two layouts are available (see LAYOUT below):
    'strips'  full-size thirds. Cue 1 uses three full-height vertical strips,
              cue 2 three full-width horizontal strips. The two centre strips
              differ overall, BUT they overlap on the central W/3 x H/3
              rectangle, and because both are cropped from the same master
              field that overlap is the identical stimulus. With
              SEGMENT_MODE='fixed' it is identical frame for frame, so the
              neurons with receptive fields in that rectangle get exactly the
              same drive in both sequences.
    'grid'    all six positions are equal-area W/3 x H/3 tiles of a 3x3 grid.
              Cue 1 is the middle row, cue 2 the middle column, so the whole
              centre tile is the same stimulus in both sequences, surround
              included.

The difference that matters is the SURROUND, not the crop. In 'strips' the
shared central neurons see identical feedforward drive but a different
surround (a tall strip above/below vs a wide strip left/right), so lateral
and feedback interactions differ. In 'grid' the surround is blank in both
cases, so the shared population is matched completely - at the cost of
driving a ninth of the screen per segment instead of a third.

The noise is a binarized version of a spatiotemporally bandpass-filtered
Gaussian noise field:
    spatial highpass corner   0.05 cycles/degree
    spatial lowpass cutoff    0.32 cycles/degree
    temporal lowpass cutoff   4 Hz

If you publish work using this script please cite the PsychoPy publications:
    Peirce, JW (2007) PsychoPy - Psychophysics software in Python.
        Journal of Neuroscience Methods, 162(1-2), 8-13.
    Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy.
        Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import absolute_import, division
from psychopy import locale_setup, sound, gui, visual, core, data, event, logging, clock, parallel
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '3.1.5'
expName = 'Kristian_noise_retinotopy'  # from the Builder filename that created this script
expInfo = {'session': '001', 'participant': ''}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='Y:\\Kristian\\Psychopy\\Kristian_noise_retinotopy.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[1366, 768], fullscr=True, screen=1,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True)
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "startup"
startupClock = core.Clock()
ISI_2 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI_2')
import json
import time
import hashlib

#fix numpy random seed
np.random.seed(12345)

# =====================================================================
# STIMULUS PARAMETERS  (edit these)
# =====================================================================
sf_corner    = 0.05   # cycles / degree - spatial HIGHPASS corner (low edge of band)
sf_cutoff    = 0.32   # cycles / degree - spatial LOWPASS cutoff  (high edge of band)
tf_cutoff    = 4.0    # Hz              - temporal LOWPASS cutoff
filter_order = 2      # Butterworth order; the three values above are the -3 dB points

stim_contrast = 1.0   # 1.0 = full black/white
epoch_dur     = 0.5   # s, duration of ONE segment of the sequence
pause_dur     = 8.0   # s, blank grey period after each complete cue
n_blocks      = 54    # repeats of the outer loop (as in the grating template)

px_per_texel  = 8     # screen pixels per noise element. Smaller = finer noise, more RAM
master_dur    = 10.0  # s of unique noise generated up front; epochs are random 1 s cuts
use_cache     = True  # cache the generated noise to stimuli/*.npy for fast restarts

# ---------------------------------------------------------------------
# LAYOUT - this is a real design decision, not a cosmetic one.
#
#   'strips'  Each cue tiles the screen with its own thirds (as before, plus
#             a middle one). Cue 1 uses three full-height vertical strips,
#             cue 2 three full-width horizontal strips. Every segment drives
#             the same amount of cortex, but the two CENTRE segments are
#             different stimuli and drive overlapping-but-not-identical
#             populations.
#
#   'grid'    A 3x3 grid of W/3 x H/3 tiles. Cue 1 is the middle row, cue 2
#             the middle column. All six positions are equal area AND the
#             centre tile is the SAME stimulus in both sequences - so you can
#             ask whether that shared population reactivates differently
#             depending on which sequence it was embedded in. Costs you
#             stimulus area, and therefore driven cortex, per segment.
# ---------------------------------------------------------------------
LAYOUT = 'strips'

# SEGMENT_MODE controls which slice of the master noise each segment plays.
#   'fixed'   every position always plays the same frames, so each repeat is
#             the identical movie. The two centre strips are locked to the
#             SAME frames, which is what makes their shared central rectangle
#             identical between cue 1 and cue 2. Use this if you want the
#             shared-element design.
#   'random'  a fresh random slice every presentation. Averages out any
#             response to a particular pattern, but the two centre strips then
#             show different noise and nothing is shared.
SEGMENT_MODE = 'fixed'

# Order within a cue is FIXED (sequential); which cue comes first is randomised.
if LAYOUT == 'strips':
    cues = [dict(cue=1, cue_name='cue1_azimuth',
                 epochs=['left', 'centre_v', 'right']),
            dict(cue=2, cue_name='cue2_elevation',
                 epochs=['top', 'centre_h', 'bottom'])]
    # distinct single bits, so every transition is unambiguous on the port
    region_ids = {'left': 1, 'right': 2, 'top': 4, 'bottom': 8,
                  'centre_v': 16, 'centre_h': 32}
elif LAYOUT == 'grid':
    cues = [dict(cue=1, cue_name='cue1_azimuth',
                 epochs=['left', 'centre', 'right']),
            dict(cue=2, cue_name='cue2_elevation',
                 epochs=['top', 'centre', 'bottom'])]
    region_ids = {'left': 1, 'right': 2, 'top': 4, 'bottom': 8, 'centre': 16}
else:
    raise ValueError("LAYOUT must be 'strips' or 'grid'")

# ---------------------------------------------------------------------
# PARALLEL PORT
#
# Only 3 NIDAQ lines are available, so the port carries the CUE, not the
# segment. One code per cue sequence, raised at the onset of segment 1 and
# held high across all three segments (1.5 s), then cleared to 0 for the
# pause. It is NOT rewritten at segment boundaries, so the line is
# genuinely continuous - no glitch for the DAQ to catch.
#
#   line 0 (value 1) = cue 1 active
#   line 1 (value 2) = cue 2 active
#   line 2 (value 4) = optional segment tick, pulsed at each segment onset
#
# Highest value sent is 2|4 = 6, so three lines is enough.
#
# Segment identity stays recoverable: the order within a cue is fixed (see
# `cues` above) and each segment is exactly epoch_dur long, so segment k
# starts at cue_onset + k*epoch_dur. Every segment is also written to the
# jsonl log and the csv with its own timestamp, so you never have to rely
# on the port alone.
# ---------------------------------------------------------------------
cue_codes    = {1: 1, 2: 2}
SEGMENT_TICK = True   # use the spare third line to mark segment onsets
TICK_BIT     = 4
TICK_FRAMES  = 2      # tick pulse width in frames (~33 ms at 60 Hz)

if SEGMENT_TICK and (max(cue_codes.values()) | TICK_BIT) > 7:
    raise ValueError('port values need more than 3 lines')

# =====================================================================
# MONITOR GEOMETRY -> pixels per degree
# =====================================================================
from psychopy import monitors
_mon = monitors.Monitor('testMonitor')
mon_width_cm = _mon.getWidth()
view_dist_cm = _mon.getDistance()
if mon_width_cm is None:
    mon_width_cm = 30.0     # fallback - SET THIS PROPERLY in the Monitor Center
    logging.warn('testMonitor width unknown, assuming %.1f cm' % mon_width_cm)
if view_dist_cm is None:
    view_dist_cm = 57.0     # fallback - SET THIS PROPERLY in the Monitor Center
    logging.warn('testMonitor distance unknown, assuming %.1f cm' % view_dist_cm)

W = int(win.size[0])
H = int(win.size[1])

# Degrees per pixel at the CENTRE of a flat screen. On a flat monitor the
# deg/pixel shrinks towards the edges, so cycles/degree is exact at the centre
# and slightly compressed in the periphery. Fine for a monitor subtending a
# modest angle; if your screen is very close to the eye, consider a spherical
# correction instead.
cm_per_px   = mon_width_cm / float(W)
deg_per_px  = np.degrees(np.arctan(cm_per_px / view_dist_cm))
ppd         = 1.0 / deg_per_px

fps = int(round(expInfo['frameRate'])) if expInfo['frameRate'] is not None else 60

# ---- noise grid ----
nx_master = int(round(W / float(px_per_texel)))
ny_master = int(round(H / float(px_per_texel)))
nt_master = int(round(master_dur * fps))
deg_per_texel_x = (W / float(nx_master)) * deg_per_px
deg_per_texel_y = (H / float(ny_master)) * deg_per_px


# ---- region geometry -----------------------------------------------
def build_regions(layout, W, H, nx, ny):
    """Return {name: {size, pos, sy, sx}} where size/pos are in screen pixels
    (centre-origin, 'pix' units) and sy/sx are slices into the master noise
    volume. Thirds are exact in both units because nx and ny are multiples
    of 3 by construction below."""
    nx3, ny3 = nx // 3, ny // 3
    fw, fh = W / 3.0, H / 3.0                      # patch size in screen px
    cx = [-W / 2.0 + (i + 0.5) * fw for i in range(3)]   # column centres, L->R
    cy = [+H / 2.0 - (j + 0.5) * fh for j in range(3)]   # row centres, T->B
    xs = [slice(i * nx3, (i + 1) * nx3) for i in range(3)]
    ys = [slice(j * ny3, (j + 1) * ny3) for j in range(3)]

    R = {}
    if layout == 'strips':
        for i, nm in enumerate(['left', 'centre_v', 'right']):
            R[nm] = dict(size=(fw, float(H)), pos=(cx[i], 0.0),
                         sy=slice(0, ny), sx=xs[i])
        for j, nm in enumerate(['top', 'centre_h', 'bottom']):
            R[nm] = dict(size=(float(W), fh), pos=(0.0, cy[j]),
                         sy=ys[j], sx=slice(0, nx))
    else:  # 'grid' - middle row is cue 1, middle column is cue 2
        R['left']   = dict(size=(fw, fh), pos=(cx[0], cy[1]), sy=ys[1], sx=xs[0])
        R['centre'] = dict(size=(fw, fh), pos=(cx[1], cy[1]), sy=ys[1], sx=xs[1])
        R['right']  = dict(size=(fw, fh), pos=(cx[2], cy[1]), sy=ys[1], sx=xs[2])
        R['top']    = dict(size=(fw, fh), pos=(cx[1], cy[0]), sy=ys[0], sx=xs[1])
        R['bottom'] = dict(size=(fw, fh), pos=(cx[1], cy[2]), sy=ys[2], sx=xs[1])
    return R


# force the texel grid to a multiple of 3 so the thirds divide exactly
nx_master = (nx_master // 3) * 3
ny_master = (ny_master // 3) * 3
deg_per_texel_x = (W / float(nx_master)) * deg_per_px
deg_per_texel_y = (H / float(ny_master)) * deg_per_px

regions = build_regions(LAYOUT, W, H, nx_master, ny_master)

# ---- sanity checks -------------------------------------------------
screen_deg_w = W * deg_per_px
screen_deg_h = H * deg_per_px
sf_nyquist = 0.5 / max(deg_per_texel_x, deg_per_texel_y)
if sf_cutoff > sf_nyquist:
    logging.error('sf_cutoff %.3f c/deg exceeds the noise-grid Nyquist %.3f c/deg. '
                  'Decrease px_per_texel.' % (sf_cutoff, sf_nyquist))
if (1.0 / sf_corner) > min(screen_deg_w, screen_deg_h):
    logging.warn('The lowest passband period (%.1f deg) is larger than the '
                 'shorter screen dimension (%.1f deg) - very few low-frequency '
                 'components will fit.' % (1.0 / sf_corner, min(screen_deg_w, screen_deg_h)))
if tf_cutoff > fps / 2.0:
    logging.error('tf_cutoff %.1f Hz exceeds the frame-rate Nyquist %.1f Hz.'
                  % (tf_cutoff, fps / 2.0))

print('screen  : %d x %d px, %.1f x %.1f deg, %.2f px/deg, %d Hz'
      % (W, H, screen_deg_w, screen_deg_h, ppd, fps))
print('noise   : %d x %d texels x %d frames, %.3f deg/texel, Nyquist %.2f c/deg'
      % (nx_master, ny_master, nt_master, deg_per_texel_x, sf_nyquist))


# =====================================================================
# NOISE GENERATION
# =====================================================================
def make_binarized_bandpass_noise(nt, ny, nx, fps, dpt_x, dpt_y,
                                  sf_low, sf_high, tf_high, order=2):
    """3D (t, y, x) Gaussian noise, Butterworth bandpass in space and
    lowpass in time, then binarized about its median.

    Spatial frequencies are handled as an isotropic radial filter, so the
    passband is an annulus in (fx, fy) between sf_low and sf_high c/deg.
    Filtering is done with an FFT, so the sequence wraps seamlessly in time
    and space - which is why we cut random 1 s segments out of it below.

    Returns int8 array of +1 / -1 with shape (nt, ny, nx).
    """
    noise = np.random.standard_normal((nt, ny, nx)).astype(np.float32)

    # frequency axes: rfftn is real along the LAST axis
    fx = np.fft.rfftfreq(nx, d=dpt_x)      # cycles / degree
    fy = np.fft.fftfreq(ny, d=dpt_y)       # cycles / degree
    ft = np.fft.fftfreq(nt, d=1.0 / fps)   # Hz

    fr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)   # radial SF, (ny, nx//2+1)
    with np.errstate(divide='ignore', invalid='ignore'):
        hp = 1.0 / np.sqrt(1.0 + (sf_low / fr) ** (2 * order))   # highpass @ sf_low
    hp[fr == 0] = 0.0                                            # kill DC
    lp = 1.0 / np.sqrt(1.0 + (fr / sf_high) ** (2 * order))      # lowpass  @ sf_high
    spatial = (hp * lp).astype(np.float32)

    temporal = (1.0 / np.sqrt(1.0 + (np.abs(ft) / tf_high) ** (2 * order))).astype(np.float32)

    F = np.fft.rfftn(noise, axes=(0, 1, 2))
    del noise
    F *= spatial[None, :, :]
    F *= temporal[:, None, None]
    filtered = np.fft.irfftn(F, s=(nt, ny, nx), axes=(0, 1, 2))
    del F

    # binarize about the median -> exactly balanced black/white
    thresh = np.median(filtered)
    binary = np.where(filtered >= thresh, 1, -1).astype(np.int8)
    return binary


# tell the operator something is happening - generation takes a few seconds
_msg = visual.TextStim(win=win, text='generating noise...', color=[0.6, 0.6, 0.6],
                       height=0.06, units='norm')
_msg.draw()
win.flip()

if not os.path.isdir('stimuli'):
    os.mkdir('stimuli')
_key = 'noise_%dx%dx%d_ppd%.4f_sf%.3f-%.3f_tf%.2f_ord%d_seed12345' % (
    nx_master, ny_master, nt_master, ppd, sf_corner, sf_cutoff, tf_cutoff, filter_order)
_cachefile = os.path.join('stimuli', hashlib.md5(_key.encode('utf-8')).hexdigest() + '.npy')

if use_cache and os.path.isfile(_cachefile):
    master_noise = np.load(_cachefile)
    print('loaded cached noise: %s' % _cachefile)
else:
    _t0 = time.time()
    master_noise = make_binarized_bandpass_noise(
        nt_master, ny_master, nx_master, fps,
        deg_per_texel_x, deg_per_texel_y,
        sf_corner, sf_cutoff, tf_cutoff, order=filter_order)
    print('generated noise in %.1f s' % (time.time() - _t0))
    if use_cache:
        np.save(_cachefile, master_noise)
        with open(_cachefile.replace('.npy', '.txt'), 'w') as f:
            f.write(_key + '\n')

# ---- crop the master field into the regions of this layout ----
# Cropping a stationary filtered field does not change its spectral content,
# so every patch has identical statistics and correct cycles/degree.
# ImageStim wants float in [-1, 1].
noise_bank = {}
for _nm, _r in regions.items():
    noise_bank[_nm] = np.ascontiguousarray(
        master_noise[:, _r['sy'], _r['sx']]).astype(np.float32)
del master_noise

region_order = cues[0]['epochs'] + [r for r in cues[1]['epochs']
                                    if r not in cues[0]['epochs']]
for _nm in region_order:
    print('  region %-9s texels %s  screen %.0fx%.0f px at (%.0f, %.0f)  code %d'
          % (_nm, noise_bank[_nm].shape[1:],
             regions[_nm]['size'][0], regions[_nm]['size'][1],
             regions[_nm]['pos'][0], regions[_nm]['pos'][1], region_ids[_nm]))

# ---- deterministic start frame per region -------------------------------
# Every region gets its own slice of the master, EXCEPT the centre regions,
# which are locked together. That lock is what guarantees the shared central
# rectangle is the same movie in both cues.
_centre_names = [n for n in region_order if n.startswith('centre')]
_others = [n for n in region_order if n not in _centre_names]
fixed_starts = {}
for _i, _nm in enumerate(_others):
    fixed_starts[_nm] = int((_i * 97) % nt_master)
_shared_start = int((len(_others) * 97) % nt_master)
for _nm in _centre_names:
    fixed_starts[_nm] = _shared_start
print('  fixed start frames: %s' % fixed_starts)

# ---- prove the shared central rectangle really is shared ----------------
if LAYOUT == 'strips':
    _nx3, _ny3 = nx_master // 3, ny_master // 3
    _sub_v = noise_bank['centre_v'][:, _ny3:2 * _ny3, :]     # its central rows
    _sub_h = noise_bank['centre_h'][:, :, _nx3:2 * _nx3]     # its central cols
    _same = np.array_equal(_sub_v, _sub_h)
    print('  central rectangle (%d x %d texels = %.0f x %.0f px) shared by both '
          'centre strips: %s' % (_sub_v.shape[2], _sub_v.shape[1],
                                 W / 3.0, H / 3.0, _same))
    if SEGMENT_MODE != 'fixed':
        print('  *** SEGMENT_MODE is %r, so the two centre strips will play '
              'DIFFERENT frames and nothing is actually shared.' % SEGMENT_MODE)

n_epoch_frames = int(round(epoch_dur * fps))

# create stimulation log file destination
if not os.path.isdir('logs'):
    os.mkdir('logs')
lt = time.localtime()
ltstr = '{0:04d}.{1:02d}{2:02d}_{3:02d}{4:02d}'.format(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min)
stimLogFile = os.path.join('logs', 'noise_{}.jsonl'.format(ltstr))
# initialize file (will overwrite existing file)
with open(stimLogFile, 'w') as f:
    pass

def writeLogEntry(dict, fd):
    """helper function to append dict to log file"""
    with open(fd, 'a+') as f:
        f.write(json.dumps(dict)+'\n')

# one header line describing the stimulus, so the log is self-documenting
writeLogEntry(dict(header=dict(expName=expName,
                               fps=fps,
                               screen_px=[W, H],
                               screen_deg=[screen_deg_w, screen_deg_h],
                               ppd=ppd,
                               mon_width_cm=mon_width_cm,
                               view_dist_cm=view_dist_cm,
                               px_per_texel=px_per_texel,
                               noise_grid=[nx_master, ny_master, nt_master],
                               deg_per_texel=[deg_per_texel_x, deg_per_texel_y],
                               sf_corner=sf_corner,
                               sf_cutoff=sf_cutoff,
                               tf_cutoff=tf_cutoff,
                               filter_order=filter_order,
                               binarized=True,
                               contrast=stim_contrast,
                               epoch_dur=epoch_dur,
                               pause_dur=pause_dur,
                               layout=LAYOUT,
                               segment_mode=SEGMENT_MODE,
                               fixed_starts=fixed_starts,
                               regions={k: dict(size=list(v['size']), pos=list(v['pos']))
                                        for k, v in regions.items()},
                               sequences={c['cue_name']: c['epochs'] for c in cues},
                               region_ids=region_ids,
                               cue_codes=cue_codes,
                               segment_tick=SEGMENT_TICK,
                               tick_bit=TICK_BIT,
                               tick_frames=TICK_FRAMES)),
              stimLogFile)

t_start = time.time()

# Initialize components for Routine "set_sequence"
set_sequenceClock = core.Clock()
nReps = len(cues)  # number of repetitions of the inner loop

# Initialize components for Routine "epoch"
epochClock = core.Clock()


noise_patch = visual.ImageStim(
    win=win, name='noise_patch', units='pix',
    image=noise_bank[cues[0]['epochs'][0]][0], mask=None,
    ori=0.0, pos=(0, 0), size=regions[cues[0]['epochs'][0]]['size'],
    color=[1,1,1], colorSpace='rgb', contrast=stim_contrast, opacity=1.0,
    texRes=128, interpolate=False, depth=-1.0)
p_port = parallel.ParallelPort(address='0x0378')
from psychopy import parallel
p_port = parallel.ParallelPort(address=0x3FF8)
p_port.setData(0)

_port_value = 0


def set_port(value):
    """Queue a port write on the next flip, but only when the value actually
    changes. This is what lets the cue code stay high across segment
    boundaries instead of being torn down and rebuilt each segment."""
    global _port_value
    value = int(value)
    if value != _port_value:
        _port_value = value
        win.callOnFlip(p_port.setData, value)



# Initialize components for Routine "pause"
pauseClock = core.Clock()
ISI_3 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI_3')

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

# ------Prepare to start Routine "startup"-------
t = 0
startupClock.reset()  # clock
frameN = -1
continueRoutine = True
routineTimer.add(1.000000)
# update component parameters for each repeat
# keep track of which components have finished
startupComponents = [ISI_2]
for thisComponent in startupComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "startup"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = startupClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # *ISI_2* period
    if t >= 0.0 and ISI_2.status == NOT_STARTED:
        # keep track of start time/frame for later
        ISI_2.tStart = t  # not accounting for scr refresh
        ISI_2.frameNStart = frameN  # exact frame index
        win.timeOnFlip(ISI_2, 'tStartRefresh')  # time at next scr refresh
        ISI_2.start(1)
    elif ISI_2.status == STARTED:  # one frame should pass before updating params and completing
        ISI_2.complete()  # finish the static period

    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in startupComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished

    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "startup"-------
for thisComponent in startupComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('ISI_2.started', ISI_2.tStart)
thisExp.addData('ISI_2.stopped', ISI_2.tStop)

# set up handler to look after randomisation of conditions etc
outer_loop = data.TrialHandler(nReps=n_blocks, method='random',
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='outer_loop')
thisExp.addLoop(outer_loop)  # add the loop to the experiment
thisOuter_loop = outer_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisOuter_loop.rgb)
if thisOuter_loop != None:
    for paramName in thisOuter_loop:
        exec('{} = thisOuter_loop[paramName]'.format(paramName))

for thisOuter_loop in outer_loop:
    currentLoop = outer_loop
    # abbreviate parameter names if possible (e.g. rgb = thisOuter_loop.rgb)
    if thisOuter_loop != None:
        for paramName in thisOuter_loop:
            exec('{} = thisOuter_loop[paramName]'.format(paramName))

    # ------Prepare to start Routine "set_sequence"-------
    t = 0
    set_sequenceClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    # shuffle the order in which the two cues are shown within this block
    trial_cues = np.random.permutation(cues).tolist()
    counter = 0
    # keep track of which components have finished
    set_sequenceComponents = []
    for thisComponent in set_sequenceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    # -------Start Routine "set_sequence"-------
    while continueRoutine:
        # get current time
        t = set_sequenceClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in set_sequenceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # -------Ending Routine "set_sequence"-------
    for thisComponent in set_sequenceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "set_sequence" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()

    # set up handler to look after randomisation of conditions etc
    inner_loop = data.TrialHandler(nReps=nReps, method='random',
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='inner_loop')
    thisExp.addLoop(inner_loop)  # add the loop to the experiment
    thisInner_loop = inner_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInner_loop.rgb)
    if thisInner_loop != None:
        for paramName in thisInner_loop:
            exec('{} = thisInner_loop[paramName]'.format(paramName))

    for thisInner_loop in inner_loop:
        currentLoop = inner_loop
        # abbreviate parameter names if possible (e.g. rgb = thisInner_loop.rgb)
        if thisInner_loop != None:
            for paramName in thisInner_loop:
                exec('{} = thisInner_loop[paramName]'.format(paramName))

        # pick this cue and step the counter
        this_cue = trial_cues[counter]
        cue_id = this_cue['cue']
        cue_name = this_cue['cue_name']
        cue_epochs = this_cue['epochs']
        counter += 1

        # set up handler for the two epochs of this cue (order is FIXED)
        epoch_loop = data.TrialHandler(nReps=len(cue_epochs), method='sequential',
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='epoch_loop')
        thisExp.addLoop(epoch_loop)  # add the loop to the experiment
        epoch_counter = 0

        for thisEpoch_loop in epoch_loop:
            currentLoop = epoch_loop

            # ------Prepare to start Routine "epoch"-------
            t = 0
            epochClock.reset()  # clock
            frameN = -1
            continueRoutine = True
            routineTimer.add(epoch_dur)
            # update component parameters for each repeat
            region = cue_epochs[epoch_counter]
            epoch_index = epoch_counter
            epoch_counter += 1
            region_id = region_ids[region]
            cue_code = cue_codes[cue_id]
            is_last_segment = (epoch_counter == len(cue_epochs))
            tick_on = False
            duration = epoch_dur

            # this epoch plays a random 1 s cut out of the master noise sequence
            seq = noise_bank[region]
            n_seq = seq.shape[0]
            if SEGMENT_MODE == 'fixed':
                start_idx = fixed_starts[region]
            else:
                start_idx = int(np.random.randint(0, n_seq))

            # write to file
            writeLogEntry(dict(noise=dict(time=time.time() - t_start,
                                          cue=cue_id,
                                          cue_name=cue_name,
                                          epoch=epoch_index,
                                          region=region,
                                          region_id=region_id,
                                          cue_code=cue_code,
                                          segment_of=len(cue_epochs),
                                          is_last_segment=is_last_segment,
                                          duration=duration,
                                          n_frames=n_epoch_frames,
                                          seq_start_frame=start_idx,
                                          segment_mode=SEGMENT_MODE,
                                          sf_corner=sf_corner,
                                          sf_cutoff=sf_cutoff,
                                          tf_cutoff=tf_cutoff,
                                          filter_order=filter_order,
                                          contrast=stim_contrast,
                                          binarized=True,
                                          )),
                         stimLogFile)

            noise_patch.setSize(regions[region]['size'], log=False)
            noise_patch.setPos(regions[region]['pos'], log=False)
            noise_patch.setImage(seq[start_idx], log=False)
            # keep track of which components have finished
            epochComponents = [noise_patch, p_port]
            for thisComponent in epochComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED

            # -------Start Routine "epoch"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = epochClock.getTime()
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame

                # *noise_patch* updates
                if t >= 0.0 and noise_patch.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    noise_patch.tStart = t  # not accounting for scr refresh
                    noise_patch.frameNStart = frameN  # exact frame index
                    win.timeOnFlip(noise_patch, 'tStartRefresh')  # time at next scr refresh
                    noise_patch.setAutoDraw(True)
                frameRemains = 0.0 + epoch_dur - win.monitorFramePeriod * 0.75  # most of one frame period left
                if noise_patch.status == STARTED and t >= frameRemains:
                    # keep track of stop time/frame for later
                    noise_patch.tStop = t  # not accounting for scr refresh
                    noise_patch.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(noise_patch, 'tStopRefresh')  # time at next scr refresh
                    noise_patch.setAutoDraw(False)
                if noise_patch.status == STARTED:  # only update if drawing
                    # advance one noise frame per screen refresh, wrapping in the bank
                    idx = (start_idx + frameN - noise_patch.frameNStart) % n_seq
                    noise_patch.setImage(seq[idx], log=False)
                # *p_port* updates - ONE code for the whole cue sequence
                if t >= 0.0 and p_port.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    p_port.tStart = t  # not accounting for scr refresh
                    p_port.frameNStart = frameN  # exact frame index
                    win.timeOnFlip(p_port, 'tStartRefresh')  # time at next scr refresh
                    p_port.status = STARTED
                    # On segment 1 this raises the cue code. On segments 2 and
                    # 3 the cue code is already high, so set_port writes
                    # nothing and the line stays continuous - only the tick
                    # bit changes.
                    set_port(cue_code | (TICK_BIT if SEGMENT_TICK else 0))
                    tick_on = SEGMENT_TICK
                if tick_on and (frameN - p_port.frameNStart) >= TICK_FRAMES:
                    set_port(cue_code)          # drop the tick, hold the cue
                    tick_on = False
                frameRemains = 0.0 + epoch_dur - win.monitorFramePeriod * 0.75  # most of one frame period left
                if p_port.status == STARTED and t >= frameRemains:
                    # keep track of stop time/frame for later
                    p_port.tStop = t  # not accounting for scr refresh
                    p_port.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(p_port, 'tStopRefresh')  # time at next scr refresh
                    p_port.status = FINISHED
                    if is_last_segment:
                        set_port(0)             # clear only at the END of the cue

                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()

                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in epochComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished

                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()

            # -------Ending Routine "epoch"-------
            for thisComponent in epochComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            epoch_loop.addData('cue', cue_id)
            epoch_loop.addData('cue_name', cue_name)
            epoch_loop.addData('epoch', epoch_index)
            epoch_loop.addData('region', region)
            epoch_loop.addData('region_id', region_id)
            epoch_loop.addData('cue_code', cue_code)
            epoch_loop.addData('seq_start_frame', start_idx)
            epoch_loop.addData('noise_patch.started', noise_patch.tStartRefresh)
            epoch_loop.addData('noise_patch.stopped', noise_patch.tStopRefresh)
            if is_last_segment and p_port.status == STARTED:
                set_port(0)
            epoch_loop.addData('p_port.started', p_port.tStart)
            epoch_loop.addData('p_port.stopped', p_port.tStop)
            thisExp.nextEntry()

        # completed both epochs of 'epoch_loop'

        # ------Prepare to start Routine "pause"-------
        t = 0
        pauseClock.reset()  # clock
        frameN = -1
        continueRoutine = True
        routineTimer.add(pause_dur)
        # update component parameters for each repeat
        # keep track of which components have finished
        pauseComponents = [ISI_3]
        for thisComponent in pauseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED

        # -------Start Routine "pause"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = pauseClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *ISI_3* period
            if t >= 0.0 and ISI_3.status == NOT_STARTED:
                # keep track of start time/frame for later
                ISI_3.tStart = t  # not accounting for scr refresh
                ISI_3.frameNStart = frameN  # exact frame index
                win.timeOnFlip(ISI_3, 'tStartRefresh')  # time at next scr refresh
                ISI_3.start(pause_dur)
            elif ISI_3.status == STARTED:  # one frame should pass before updating params and completing
                ISI_3.complete()  # finish the static period

            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()

            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pauseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished

            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()

        # -------Ending Routine "pause"-------
        for thisComponent in pauseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        inner_loop.addData('ISI_3.started', ISI_3.tStart)
        inner_loop.addData('ISI_3.stopped', ISI_3.tStop)
        thisExp.nextEntry()

    # completed nReps repeats of 'inner_loop'

    thisExp.nextEntry()

# completed n_blocks repeats of 'outer_loop'

p_port.setData(0)


# Flip one final time so any remaining win.callOnFlip()
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
