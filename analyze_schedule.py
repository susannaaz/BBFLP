# Nothing new here, this script is imported from:
# https://github.com/simonsobs/pwg-scripts/blob/24a8c8202e2f80fb9b5097ee0e2dcfe5c1c07114/pwg-tds/mbs-noise-sims-sat/analyze_schedule.py

import numpy as np
import os
import sys

import ephem
import dateutil.parser

import healpy as hp
from toast import qarray as qa


xaxis, yaxis, zaxis = np.eye(3)

fname = sys.argv[1]

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

print('Starting to analyse schedule')

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
with open(fname, "r") as schedule:
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
        "  {:>40} : {:6.2f} days ({:6.2f}%), {:4} scans ({:6.2f}%) "
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