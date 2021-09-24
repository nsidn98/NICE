python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 50 3 -r 2 -c 2 > constrained_experiment/delayed_50_buffer_exp.txt &&
python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 25 3 -r 2 -c 2 > constrained_experiment/delayed_25_buffer_exp.txt &&
python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 75 3 -r 2 -c 2 > constrained_experiment/delayed_75_buffer_exp.txt &&
python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 100 3 -r 2 -c 2 > constrained_experiment/delayed_100_buffer_exp.txt
