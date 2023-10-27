tele='SAT'
band=90
band_name='f090'
wafer=w25
schedule='schedules/schedule_sat'
atmosphere='atmosphere.toml'
beam_file='SAT_f090_beam.p'

mpirun -np 8 toast_so_sim.py \
       `# Inputs` \
       --config params.toml \
       --scan_map.file /mnt/so1/users/keskital/cmb_SAT_${band}.fits \
       --scan_map.enable \
       --job_group_size 8 \
       `# Scan params` \
       --wafer_slots ${wafer} \
       --bands SAT_${band_name} \
       --schedule ${schedule} \
       #--sim_ground.scan_rate_az "1.0 deg / s" \
       --sample_rate 37 \
       --out output_${band}_${wafer} \ 
       `# Simulation elements` \
       --sim_noise.disable \
       #--det_pointing_azel.shared_flag_mask 0 \
       #--config ${atmosphere} \
       --sim_atmosphere_coarse.disable \
       --sim_atmosphere.disable \
       #--elevation_model.disable \
       `# Outputs` \
       --save_hdf5.enable \
       --mapmaker.disable \
       --filterbin.disable \
       `# HWP parameters` \
       #--sim_ground.hwp_angle 'hwp_angle'\
       #--sim_ground.hwp_rpm 120\
       #--weights_azel.hwp_angle 'hwp_angle'\
       `# SSO simulation` \
       #--sim_sso.enable \
       #  --sim_sso.beam_file ${beam_file} \
       #--sim_sso.sso_name "Jupiter" \
