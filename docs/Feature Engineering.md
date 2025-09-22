# Determining which features are important to which tasks

Testing with 24 training envs

### Simple hover to [0, 0, 0.5] (sorted by importance):
- rel_z           (idx 15): 1.4095
- vel_z           (idx  9): 1.1828
- norm_z          (idx 18): 0.8985
- dist_to_target  (idx 19): 0.8769
- prev_ctrl3      (idx 26): 0.8663
- err_z           (idx 22): 0.7967
- pos_z           (idx  2): 0.7426
- prev_ctrl2      (idx 25): 0.5423
- quat_x          (idx  3): 0.5381
- prev_ctrl0      (idx 23): 0.5115
- ang_x           (idx 10): 0.4378
- err_x           (idx 20): 0.4111
- rel_x           (idx 13): 0.3818
- rel_y           (idx 14): 0.3627
- err_y           (idx 21): 0.2960
- norm_x          (idx 16): 0.2609
- prev_ctrl1      (idx 24): 0.2084
- pos_y           (idx  1): 0.2030
- vel_y           (idx  8): 0.1755
- ang_z           (idx 12): 0.1287
- pos_x           (idx  0): 0.1238
- quat_y          (idx  4): 0.1147
- norm_y          (idx 17): 0.1142
- quat_z          (idx  5): 0.0962
- quat_w          (idx  6): 0.0202
- ang_y           (idx 11): 0.0194
- vel_x           (idx  7): 0.0083

Results seem to make sense (high importance on features correlated with Z height and distance). It appears prev_ctrl3 (pitch) does show relatively high which is strange.

Update 1: Added derivative (err(t) - err(t-1)) and integral (running sum) of position error
- integral_err_z, derivative_err_z, sec_derivative_err_z
- Sample results
  - err_z           (idx 22): 1.2179
  - dist_to_target  (idx 19): 0.8704
  - rel_z           (idx 15): 0.8028
  - integral_err_z  (idx 31): 0.6798
  - vel_z           (idx  9): 0.5233
  - sec_derivative_err_z (idx 28): 0.5006
  - quat_x          (idx  3): 0.4597
  - derivative_err_z (idx 25): 0.4315
  - ...
- Results: No more jitter after stabilizing, still stabilizes at incorrect Z for low/high target Z