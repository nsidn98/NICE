python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 25 3 -r 2 > robustness_experiments/delayed_25_buffer_exp.txt &&
python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 50 3 -r 2 > robustness_experiments/delayed_50_buffer_exp.txt &&
python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 75 3 -r 2 > robustness_experiments/delayed_75_buffer_exp.txt &&
python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 100 3 -r 2 > robustness_experiments/delayed_100_buffer_exp.txt &&

python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d2_m1.0_b0.0.ckpt -d flights_delay -n 25 3 -r 2 > robustness_experiments/delayed_25_moveup_exp.txt &&
python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d2_m1.0_b0.0.ckpt -d flights_delay -n 50 3 -r 2 > robustness_experiments/delayed_50_moveup_exp.txt &&
python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d2_m1.0_b0.0.ckpt -d flights_delay -n 75 3 -r 2 > robustness_experiments/delayed_75_moveup_exp.txt &&
python3 test_disrupt.py -l feasibility -i 100 -p model_wts/s0_d2_m1.0_b0.0.ckpt -d flights_delay -n 100 3 -r 2 > robustness_experiments/delayed_100_moveup_exp.txt &&

python3 test_disrupt.py -l moveup -i 100 -p model_wts/s0_d2_m1.0_b0.0.ckpt -d flights_delay -n 25 3 -r 2 > robustness_experiments/delayed_25_moveup_exp_vMoveup.txt &&
python3 test_disrupt.py -l moveup -i 100 -p model_wts/s0_d2_m1.0_b0.0.ckpt -d flights_delay -n 50 3 -r 2 > robustness_experiments/delayed_50_moveup_exp_vMoveup.txt &&
python3 test_disrupt.py -l moveup -i 100 -p model_wts/s0_d2_m1.0_b0.0.ckpt -d flights_delay -n 75 3 -r 2 > robustness_experiments/delayed_75_moveup_exp_vMoveup.txt &&
python3 test_disrupt.py -l moveup -i 100 -p model_wts/s0_d2_m1.0_b0.0.ckpt -d flights_delay -n 100 3 -r 2 > robustness_experiments/delayed_100_moveup_exp_vMoveup.txt &&

python3 test_disrupt.py -l buffer -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 25 3 -r 2 > robustness_experiments/delayed_25_buffer_exp_vBuffer.txt &&
python3 test_disrupt.py -l buffer -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 50 3 -r 2 > robustness_experiments/delayed_50_buffer_exp_vBuffer.txt &&
python3 test_disrupt.py -l buffer -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 75 3 -r 2 > robustness_experiments/delayed_75_buffer_exp_vBuffer.txt &&
python3 test_disrupt.py -l buffer -i 100 -p model_wts/s0_d1_m0.0_b1.0.ckpt -d flights_delay -n 100 3 -r 2 > robustness_experiments/delayed_100_buffer_exp_vBuffer.txt