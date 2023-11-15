#!/bin/bash

# Example: Run 1 day observation in the south patch:
toast_ground_schedule \
    @schedules/schedule_sat.par \
    @schedules/patches_sat.txt --debug
