#!/bin/bash

module use --append /scratch/gpfs/ip8725/conda_envs/rp_env_0.0.5.dev84/modulefiles
module load soconda/0.0.5.dev84

# Equivalent to
# toast_ground_schedule schedule_sat.baseline.par
# But customising timescales, observation patch, etc.
# 
# Example: Run observation in the south patch:
toast_ground_schedule \
    @schedules/schedule_sat.par \
    @schedules/patches_sat.txt --debug
