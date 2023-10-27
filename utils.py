import numpy as np
import datetime as dt
import copy
import argparse
import os
import sys
import traceback
import healpy as hp
from astropy import units as u
import warnings
warnings.simplefilter("ignore")
#
# Preprocess branch of sotodlib
from sotodlib import coords, core
import so3g
import yaml
from sotodlib.core import Context
from sotodlib.hwp import hwp
from sotodlib.tod_ops import fft_ops
from sotodlib.tod_ops.fft_ops import calc_psd
import logging
#
# Use sotodlib.toast to set default 
# object names used in toast
import sotodlib.toast as sotoast
import toast
import toast.ops
from toast.mpi import MPI, Comm
from toast import spt3g as t3g
if t3g.available:
    from spt3g import core as c3g
import sotodlib.toast.ops as so_ops
import sotodlib.mapmaking
# Import pixell
import pixell.fft
pixell.fft.engine = "fftw"
#
# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

# A collection of useful functions written or modified by SA and other SO members, including Max.

def make_aman(sch, fig, ax, print_info=False, make_plot=False):
    DEG = np.pi/180
    ts_st = dt.datetime.strptime(sch[0]+' '+sch[1], '%Y-%m-%d %H:%M:%S').timestamp()
    ts_en = dt.datetime.strptime(sch[2]+' '+sch[3], '%Y-%m-%d %H:%M:%S').timestamp()
    if print_info:
        print(dt.datetime.strptime(sch[0]+' '+sch[1], '%Y-%m-%d %H:%M:%S'),
              dt.datetime.strptime(sch[2]+' '+sch[3], '%Y-%m-%d %H:%M:%S'),
              (ts_en-ts_st)/60, sch[6], sch[7])
    
    names = ['a']
    ts = np.arange(ts_st, ts_en+0.5, 0.5)
    sig = np.ones((1,len(ts)),dtype='float32')
    aman = core.AxisManager().wrap('signal', sig, [(0, core.LabelAxis('dets', names)),
                                               (1, core.IndexAxis('samps'))])
    aman.wrap('timestamps', ts, [(0, core.IndexAxis('samps')),])
    
    focalplane = core.AxisManager().wrap('xi', np.zeros(1), [(0, core.LabelAxis('dets', names)),])
    focalplane.wrap('eta', np.zeros(1), [(0, core.LabelAxis('dets', names)),])
    focalplane.wrap('gamma', np.zeros(1), [(0, core.LabelAxis('dets', names)),])
    aman.wrap('focal_plane', focalplane)

    RL_scan = (((ts - ts[0])//(sch[7]-sch[6]))%2).astype(bool)
    ndeg = (ts - ts[0])%(sch[7]-sch[6])
    Az_scan = [sch[7] - nd if RL_scan[i] else sch[6] + nd for i, nd in enumerate(ndeg)]
    el = np.ones(len(Az_scan))*sch[8]
    rot = np.ones(len(Az_scan))*sch[4]
    
    boresite = core.AxisManager().wrap('az', np.asarray(Az_scan)*DEG, [(0, core.IndexAxis('samps')),])
    boresite.wrap('el', el*DEG, [(0, core.IndexAxis('samps')),])
    boresite.wrap('roll', rot*DEG, [(0, core.IndexAxis('samps')),])
    aman.wrap('boresight', boresite)
    
    csl = so3g.proj.CelestialSightLine.az_el(aman.timestamps, aman.boresight['az'], aman.boresight['el'],
                                             site='so', weather='typical')
    
    boresiteradec = core.AxisManager().wrap('ra', np.rad2deg(csl.coords()[:,0]), 
                                            [(0, core.IndexAxis('samps')),])
    boresiteradec.wrap('dec', np.rad2deg(csl.coords()[:,1]), 
                       [(0, core.IndexAxis('samps')),])
    aman.wrap('boresight_radec', boresiteradec)
    if make_plot:
        ax[0].plot(np.rad2deg(csl.coords()[:,0]), np.rad2deg(csl.coords()[:,1]),'.')
        ax[0].set_xlabel('RA')
        ax[0].set_ylabel('DEC')

        ax[1].plot(aman.timestamps,Az_scan)
        ax[1].set_xlabel('Timestamp [s]')
        ax[1].set_ylabel('Az')

        ax[2].plot(aman.timestamps, aman.boresight.el)
        ax[2].set_xlabel('Timestamp [s]')
        ax[2].set_ylabel('El')

        plt.tight_layout()
    ra_cen = np.mean(np.rad2deg(csl.coords()[:,0]))
    dec_cen = np.mean(np.rad2deg(csl.coords()[:,1]))
    return ra_cen, dec_cen, aman


def plot_one_timestream(data):
    obs = data.obs[0]
    dets = obs.local_detectors

    fig, ax = plt.subplots(1,1,figsize=(6,1))
    ax.plot(obs.shared["times"], obs.detdata["signal"][dets[0]])
    ax.set_xlabel('$UTC$')
    ax.set_ylabel('$K_{cmb}$')

def compare(func):
    def plot(*args):
        # Get data 
        data_before = args[0]
        obs_before = data_before.obs[0]
        dets_before = obs_before.local_detectors
        time_before = copy.deepcopy(obs_before.shared["times"])
        signal_before = copy.deepcopy(obs_before.detdata["signal"][dets_before[0]])
        # Data after applying the function to first detector
        data_after, operator = func(*args)
        obs_after = data_after.obs[0]
        dets_after = obs_after.local_detectors
        time_after = copy.deepcopy(obs_after.shared["times"])
        signal_after = copy.deepcopy(obs_after.detdata["signal"][dets_after[0]])
        
        # Plot before and after 
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6,2), sharex=True)
        ax[0].plot(time_before, signal_before, label='(before)', alpha=0.5)
        ax[0].plot(time_after, signal_after, label='(after)', alpha=0.5)
        ax[0].set_ylabel('Signal [$K_{cmb}$]')
        ax[0].legend()
        # plot difference (after - before)
        ax[1].plot(time_after, signal_after - signal_before,
                   label='(After - Before)')
        ax[1].set_xlabel('$UTC$')
        ax[1].set_ylabel('Diff')
        ax[1].legend()
        return data_after, operator

    return plot

def apply_scanning(data, telescope, schedule):
    sim_gnd = toast.ops.SimGround(
            telescope=telescope,
            schedule=schedule,
            weather="atacama",
            hwp_angle='hwp_angle',
            hwp_rpm=120,
        )
    sim_gnd.apply(data)
    return data, sim_gnd

def apply_det_pointing_radec(data, sim_gnd):
    det_pointing_radec =toast.ops.PointingDetectorSimple(name='det_pointing_radec', quats='quats_radex', shared_flags=None)
    det_pointing_radec.boresight = sim_gnd.boresight_radec
    det_pointing_radec.apply(data)
    return data, det_pointing_radec

def apply_det_pointing_azel(data, sim_gnd):
    det_pointing_azel =toast.ops.PointingDetectorSimple(name='det_pointing_azel', quats='quats_azel', shared_flags=None)
    det_pointing_azel.boresight = sim_gnd.boresight_azel
    det_pointing_azel.apply(data)
    return data, det_pointing_azel

def apply_pixels_radec(data, det_pointing_radec, nside):
    pixels_radec = toast.ops.pixels_healpix.PixelsHealpix(
        name="pixels_radec", 
        pixels='pixels',
        nside=nside,
        nside_submap=8,)
    pixels_radec.detector_pointing = det_pointing_radec
    pixels_radec.apply(data)
    return data, pixels_radec

def apply_weights_radec(data, det_pointing_radec):
    weights_radec = toast.ops.stokes_weights.StokesWeights(
        mode = "IQU", # The Stokes weights to generate (I or IQU)
        name = "weights_radec", # The 'name' of this class instance,
        hwp_angle = "hwp_angle"
    )
    weights_radec.detector_pointing = det_pointing_radec
    weights_radec.apply(data)
    return data, weights_radec

@compare
def apply_scan_map(data, file, pixels_radec, weights_radec):
    scan_map = toast.ops.ScanHealpixMap(
        name='scan_map',
        file=file)
    scan_map.pixel_pointing= pixels_radec
    scan_map.stokes_weights = weights_radec
    scan_map.apply(data)
    return data, scan_map

@compare
def apply_noise_model(data):
    noise_model = toast.ops.DefaultNoiseModel(
        name='default_model', noise_model='noise_model')
    noise_model.apply(data)
    return data, noise_model

@compare
def apply_sim_noise(data):
    sim_noise = toast.ops.SimNoise()
    sim_noise.apply(data)
    return data, sim_noise