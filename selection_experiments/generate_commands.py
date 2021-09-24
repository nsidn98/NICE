# python3 test_disrupt.py -l feasibility -i 100 -s 1357535170 -p model_wts/original_buffer_8.ckpt -d flights_delay -n 50 3 > robustness_experiments/delayed_50_buffer_exp.txt &

commands = []
for seed in range(5):
    for density in range(1, 4):
        subcommands = []
        for n in [0, 2, 4, 8]:
            for buffer in [0, 1]:
                network = f"s{seed}_d{density}_m{1 - buffer}.0_b{buffer}.0"
                outfile = f"s{seed}_d{density}_m{1 - buffer}_b{buffer}_n{n}"
                command = ("python3 test_disrupt.py "
                           "-l feasibility "
                           "-i 20 "
                           f"-p model_wts/{network}.ckpt "
                           "-d flights_delay " 
                           "-n 50 3 "
                           f"-r {n} "
                           f"> selection_experiments/{outfile}.txt")
                commands.append(command)
with open("selection_experiments/run_exps.sh", "w+") as f:
    f.write(" &&\n".join(commands))

                