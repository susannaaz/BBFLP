#!/bin/bash

module use --append /scratch/gpfs/ip8725/conda_envs/rp_env_0.0.5.dev84/modulefiles
module load soconda/0.0.5.dev84

# Equivalent to
# toast_ground_schedule schedule.sat.baseline.par
# But customising timescales, observation patch, etc.
# 
# Example: Run observation in the south patch:
toast_ground_schedule --block-out 2025/07/01-2025/07/01 \
    --equalize-area \
    --equalize-time \
    --site-lat -22.958064 \
    --site-lon -67.786222 \
    --site-alt 5200 \
    --site-name ATACAMA \
    --telescope SAT \
    --patch south,0.001,-50.000,-30.000,90.000,-50.000 --debug \
    --patch-coord C \
    --el-min 55 \
    --el-max 70 \
    --sun-el-max 90 \
    --sun-avoidance-angle 45 \
    --moon-avoidance-angle 45 \
    --start "2025-07-01 00:00:00" \
    --stop "2025-07-01 00:10:00" \
    --gap-s 60 \
    --gap-small 0 \
    --ces-max-time 1200 \
    --fp-radius 0 \
    --out schedules/schedule_sat_10min.txt \
    --ra-period 5 \
    --ra-amplitude 5 \
    --boresight-angle-step 45 \
    --boresight-angle-time 1440 \
    --elevations-deg 55 

