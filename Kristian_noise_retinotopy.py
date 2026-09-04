#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Binarized bandpass-filtered noise, presented in screen thirds.

Built from the Kristian_functional_190909.py PsychoPy3 Builder template
(v3.1.5) - same window setup, same loop scaffolding, same parallel-port
component format.

Instead of drifting gratings this presents sequences of noise:
    Cue 1 : noise on the far LEFT third   (1 s), then the far RIGHT third  (1 s)
    Cue 2 : noise on the TOP third        (1 s), then the BOTTOM third     (1 s)

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
epoch_dur     = 1.0   # s, duration of ONE noise presentation (one screen third)
pause_dur     = 8.0   # s, blank grey period after each complete cue
n_blocks      = 54    # repeats of the outer loop (as in the grating template)

px_per_texel  = 8     # screen pixels per noise element. Smaller = finer noise, more RAM
master_dur    = 10.0  # s of unique noise generated up front; epochs are random 1 s cuts
use_cache     = True  # cache the generated noise to stimuli/*.npy for fast restarts

# Parallel port codes - all distinct single bits so transitions are unambiguous
port_codes = {'left': 1, 'right': 2, 'top': 4, 'bottom': 8}

# The two cues. Order within a cue is fixed (sequential); cue order is randomised.
cues = [dict(cue=1, cue_name='cue1_horizontal', epochs=['left', 'right']),
        dict(cue=2, cue_name='cue2_vertical',   epochs=['top', 'bottom'])]

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

# ---- screen third geometry, in pixels (centre-origin, 'pix' units) ----
W3 = W // 3
H3 = H // 3
patch_size_pix = {'left':  (W3, H), 'right':  (W3, H),
                  'top':   (W, H3), 'bottom': (W, H3)}
patch_pos_pix  = {'left':   (-(W / 2.0 - W3 / 2.0), 0.0),
                  'right':  (+(W / 2.0 - W3 / 2.0), 0.0),
                  'top':    (0.0, +(H / 2.0 - H3 / 2.0)),
                  'bottom': (0.0, -(H / 2.0 - H3 / 2.0))}

# ---- noise grid ----
nx_master = int(round(W / float(px_per_texel)))
ny_master = int(round(H / float(px_per_texel)))
nt_master = int(round(master_dur * fps))
deg_per_texel_x = (W / float(nx_master)) * deg_per_px
deg_per_texel_y = (H / float(ny_master)) * deg_per_px

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

# ---- crop the master field into the four screen thirds ----
# Cropping a stationary filtered field does not change its spectral content,
# so every patch has identical statistics and correct cycles/degree.
nx3 = nx_master // 3
ny3 = ny_master // 3
noise_bank = {
    'left':   np.ascontiguousarray(master_noise[:, :, :nx3]),
    'right':  np.ascontiguousarray(master_noise[:, :, nx_master - nx3:]),
    'top':    np.ascontiguousarray(master_noise[:, :ny3, :]),
    'bottom': np.ascontiguousarray(master_noise[:, ny_master - ny3:, :]),
}
# ImageStim wants float in [-1, 1]
for _k in noise_bank:
    noise_bank[_k] = noise_bank[_k].astype(np.float32)
del master_noise

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
                               port_codes=port_codes)),
              stimLogFile)

t_start = time.time()

# Initialize components for Routine "set_sequence"
set_sequenceClock = core.Clock()
nReps = len(cues)  # number of repetitions of the inner loop

# Initialize components for Routine "epoch"
epochClock = core.Clock()


noise_patch = visual.ImageStim(
    win=win, name='noise_patch', units='pix',
    image=noise_bank['left'][0], mask=None,
    ori=0.0, pos=(0, 0), size=patch_size_pix['left'],
    color=[1,1,1], colorSpace='rgb', contrast=stim_contrast, opacity=1.0,
    texRes=128, interpolate=False, depth=-1.0)
p_port = parallel.ParallelPort(address='0x0378')
from psychopy import parallel
p_port = parallel.ParallelPort(address=0x3FF8)
p_port.setData(0)



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
            code = port_codes[region]
            duration = epoch_dur

            # this epoch plays a random 1 s cut out of the master noise sequence
            seq = noise_bank[region]
            n_seq = seq.shape[0]
            start_idx = int(np.random.randint(0, n_seq))

            # write to file
            writeLogEntry(dict(noise=dict(time=time.time() - t_start,
                                          cue=cue_id,
                                          cue_name=cue_name,
                                          epoch=epoch_index,
                                          region=region,
                                          code=code,
                                          duration=duration,
                                          n_frames=n_epoch_frames,
                                          seq_start_frame=start_idx,
                                          sf_corner=sf_corner,
                                          sf_cutoff=sf_cutoff,
                                          tf_cutoff=tf_cutoff,
                                          filter_order=filter_order,
                                          contrast=stim_contrast,
                                          binarized=True,
                                          )),
                         stimLogFile)

            noise_patch.setSize(patch_size_pix[region], log=False)
            noise_patch.setPos(patch_pos_pix[region], log=False)
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
                # *p_port* updates
                if t >= 0.0 and p_port.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    p_port.tStart = t  # not accounting for scr refresh
                    p_port.frameNStart = frameN  # exact frame index
                    win.timeOnFlip(p_port, 'tStartRefresh')  # time at next scr refresh
                    p_port.status = STARTED
                    win.callOnFlip(p_port.setData, int(code))
                frameRemains = 0.0 + epoch_dur - win.monitorFramePeriod * 0.75  # most of one frame period left
                if p_port.status == STARTED and t >= frameRemains:
                    # keep track of stop time/frame for later
                    p_port.tStop = t  # not accounting for scr refresh
                    p_port.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(p_port, 'tStopRefresh')  # time at next scr refresh
                    p_port.status = FINISHED
                    win.callOnFlip(p_port.setData, int(0))

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
            epoch_loop.addData('code', code)
            epoch_loop.addData('seq_start_frame', start_idx)
            epoch_loop.addData('noise_patch.started', noise_patch.tStartRefresh)
            epoch_loop.addData('noise_patch.stopped', noise_patch.tStopRefresh)
            if p_port.status == STARTED:
                win.callOnFlip(p_port.setData, int(0))
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
