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

# Preprocess branch of sotodlib
from sotodlib import coords, core
import so3g
import yaml
from sotodlib.core import Context
from sotodlib.hwp import hwp
from sotodlib.tod_ops import fft_ops
from sotodlib.tod_ops.fft_ops import calc_psd
import logging
from sotodlib.site_pipeline.preprocess_tod import _build_pipe_from_configs, _get_preprocess_context
from sotodlib.coords import demod
from sotodlib import coords

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

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

# For analysis of schedule
import ephem
import dateutil.parser
import healpy as hp
from toast import qarray as qa
import subprocess


'''
A collection of useful functions written by SA, and/or other SO members.
'''

def run_bash_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Command output:", result.stdout)
        print("Command error (if any):", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error running command:", e)
        
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


def analyze_schedule(schedule_file):
    '''
    Modified version of:
    https://github.com/simonsobs/pwg-scripts/blob/24a8c8202e2f80fb9b5097ee0e2dcfe5c1c07114/pwg-tds/mbs-noise-sims-sat/analyze_schedule.py
    '''
    xaxis, yaxis, zaxis = np.eye(3)

    sun_avoidance = 45
    moon_avoidance = 45


    class Patch(object):
        time = 0
        count = 0
        rising_count = 0
        setting_count = 0

        def __init__(self, name):
            self.name = name
            self.elevations = []

    def MJD_to_DJD(mjd):
        return mjd + 2400000.5 - 2415020

    patches = {}

    def from_angles(az, el):
        elquat = qa.rotation(yaxis, np.radians(90 - el))
        azquat = qa.rotation(zaxis, np.radians(az))
        return qa.mult(azquat, elquat)

    def unwind(quat1, quat2):
        if np.sum(np.abs(quat1 - quat2)) > np.sum(np.abs(quat1 + quat2)):
            return -quat2
        else:
            return quat2

    def at_closest(az1, az2, el, sun_az1, sun_el1, sun_az2, sun_el2):
        if az2 < az1:
            az2 += 360
        naz = max(3, int((az2 - az1)))
        quats = []
        for az in np.linspace(az1, az2, naz):
            quats.append(from_angles(az % 360, el))
        sun_quat1 = from_angles(sun_az1, sun_el1)
        sun_quat2 = from_angles(sun_az2, sun_el2)
        sun_quat2 = unwind(sun_quat1, sun_quat2)
        t = np.linspace(0, 1, 10)
        sun_quats = qa.slerp(t, [0, 1], [sun_quat1, sun_quat2])
        vecs = qa.rotate(quats, zaxis)
        sun_vecs = qa.rotate(sun_quats, zaxis).T
        dpmax = -1
        for vec in vecs:
            dps = np.dot(vec, sun_vecs)
            dpmax = max(-1, np.amax(dps))
        min_dist = np.degrees(np.arccos(dpmax))
        return min_dist

    def check_sso(observer, az1, az2, el, sso, angle, mjdstart, mjdstop):
        """ Determine if the solar system object (SSO) enters the scan.
        """
        if az2 < az1:
            az2 += 360
        naz = max(3, int(0.25 * (az2 - az1) * np.cos(np.radians(el))))
        quats = []
        for az in np.linspace(az1, az2, naz):
            quats.append(from_angles(az % 360, el))
        vecs = qa.rotate(quats, zaxis)

        tstart = MJD_to_DJD(mjdstart)
        tstop = MJD_to_DJD(mjdstop)
        t1 = tstart
        # Test every hour separately
        while t1 < tstop:
            t2 = min(tstop, t1 + 1 / 24)
            observer.date = t1
            sso.compute(observer)
            sun_az1, sun_el1 = np.degrees(sso.az), np.degrees(sso.alt)
            observer.date = t2
            sso.compute(observer)
            sun_az2, sun_el2 = np.degrees(sso.az), np.degrees(sso.alt)
            sun_quat1 = from_angles(sun_az1, sun_el1)
            sun_quat2 = from_angles(sun_az2, sun_el2)
            sun_quat2 = unwind(sun_quat1, sun_quat2)
            t = np.linspace(0, 1, 10)
            sun_quats = qa.slerp(t, [0, 1], [sun_quat1, sun_quat2])
            sun_vecs = qa.rotate(sun_quats, zaxis).T
            dpmax = np.amax(np.dot(vecs, sun_vecs))
            min_dist = np.degrees(np.arccos(dpmax))
            if min_dist < angle:
                return True
            t1 = t2
        return False

    total_start = None
    total_count = 0
    total_time = 0
    sun_count = 0
    sun_time = 0
    moon_count = 0
    moon_time = 0
    header = None
    sun = ephem.Sun()
    moon = ephem.Moon()
    el_time = np.zeros(90)
    with open(schedule_file, "r") as schedule:
        for iline, line in enumerate(schedule):
            if iline == 1:
                site, telescope, site_lat, site_lon, site_alt = line.split()
                observer = ephem.Observer()
                observer.lon = site_lon
                observer.lat = site_lat
                observer.elevation = float(site_alt)  # In meters
                observer.epoch = "2000"
                observer.temp = 0  # in Celsius
                observer.compute_pressure()
            if line.startswith("#"):
                header = line[:-1]
                continue
            parts = line.split()
            if len(parts) != 11:
                continue
            # print(line)
            name = parts[5]
            rising = float(parts[6]) < 180
            start_time = f"{parts[0]} {parts[1]} +0000"
            stop_time = f"{parts[2]} {parts[3]} +0000"
            start_utc = dateutil.parser.parse(start_time)
            stop_utc = dateutil.parser.parse(stop_time)
            mjd_start = start_utc.timestamp() / 86400.0 + 2440587.5 - 2400000.5
            mjd_stop = stop_utc.timestamp() / 86400.0 + 2440587.5 - 2400000.5
            start = mjd_start
            if total_start is None:
                total_start = start
            stop = mjd_stop
            sub = int(parts[-1])
            if name not in patches:
                patches[name] = Patch(name)
            patch = patches[name]
            patch.time += stop - start
            total_time += stop - start
            if sub == 0:
                patch.count += 1
                if rising:
                    patch.rising_count += 1
                else:
                    patch.setting_count += 1
                total_count += 1
            az1 = float(parts[6])
            az2 = float(parts[7])
            el = float(parts[8])
            el_time[int(el)] += stop - start
            patch.elevations.append(el)
            if check_sso(observer, az1, az2, el, sun, sun_avoidance, mjd_start, mjd_stop):
                print("Sun too close on line # {}!".format(iline))
                print(header)
                print(line)
                sun_time += stop - start
                sun_count += 1
            if check_sso(observer, az1, az2, el, moon, moon_avoidance, mjd_start, mjd_stop):
                print("Moon too close on line # {}!".format(iline))
                print(header)
                print(line)
                moon_time += stop - start
                moon_count += 1

    total_stop = stop

    available_time = total_stop - total_start
    print(
        "Total time: {:.2f} days. Scheduled time: {:.2f} days "
        "({:.2f}% efficiency), {} scans".format(
            available_time, total_time, total_time * 100 / available_time, total_count
        )
    )

    print(
        "Compromised by Sun: {:.2f} days "
        "({:.2f}%), {} scans".format(sun_time, sun_time * 100 / total_time, sun_count)
    )

    print(
        "Compromised by Moon: {:.2f} days "
        "({:.2f}%), {} scans".format(moon_time, moon_time * 100 / total_time, moon_count)
    )

    for name in sorted(patches.keys()):
        patch = patches[name]
        els = np.array(patch.elevations)
        print(
            "{:>40} : {:6.2f} days ({:6.2f}%), {:4} scans ({:6.2f}%) "
            "{:6.2f}% rising. "
            "El: {:5.1f} < {:5.1f} +- {:5.1f} < {:5.1f}".format(
                name,
                patch.time,
                patch.time * 100 / total_time,
                patch.count,
                patch.count * 100 / total_count,
                patch.rising_count * 100 / patch.count,
                np.amin(els),
                np.mean(els),
                np.std(els),
                np.amax(els),
            )
        )

    print("Cumulative observing time by elevation")

    ctime = 0
    for el in range(90):
        if el_time[el] == 0:
            continue
        ctime += el_time[el]
        print("el < {} deg: {:6.3f} %".format(el + 1, 100 * ctime / total_time))
        
    return
        

def split_schedule(schedule_file):
    ''' This function processes a schedule containing 
        numerous observations and splits it into
        individual observation files. 
    '''
    
    # Read the schedule file and count the number of lines
    with open(schedule_file, 'r') as file:
        lines = file.readlines()
    nline = len(lines) - 3
    
    print(f"Found {nline} entries in {schedule_file}")
    
    # Split the schedule into header and body
    header = lines[:3]
    body = lines[3:]
    
    # Create the output directory
    outdir = 'schedules/split_schedule'
    os.makedirs(outdir, exist_ok=True)
    
    # Split the body into individual observation files
    for i, line in enumerate(body):
        observation_file = f"{outdir}/schedule_{i:03d}.txt"
        with open(observation_file, 'w') as file:
            file.writelines(header)
            file.write(line)
            
    return


def preprocess(aman):
    # Let's load the config file created 
    configs="./preprocess/pipe_s0002_sim_preprocess.yaml"
    configs = yaml.safe_load(open(configs, "r"))
    # And build the pipeline including the filters 
    # specified in the config
    pipe = _build_pipe_from_configs(configs)
    proc_aman = core.AxisManager(aman.dets, aman.samps)
    for pi in np.arange(3):
        # 1) Detrended
        # 2) Apodize
        # 3) Demodulate
        process = pipe[pi]
        #print(f"Processing {process.name}")
        process.process(aman, proc_aman)
        process.calc_and_save(aman, proc_aman)
    return aman, proc_aman
