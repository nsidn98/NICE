

outfiles = []
for seed in range(5):
    seed_outfiles = []
    for density in range(1, 4):
        subcommands = []
        for buffer in [1, 0]:
            for n in [0, 2, 4, 8]:
                outfile = f"s{seed}_d{density}_m{1 - buffer}_b{buffer}_n{n}.txt"
                seed_outfiles.append(outfile)
    outfiles.append(seed_outfiles)

output = []
for seed_outfiles in outfiles:
    values = []
    for outfile in seed_outfiles:
        with open(outfile) as f:
            extract = False
            for line in f.readlines():
                if extract:
                    nums = line.split(" ")
                    LP = float(nums[1])
                    NICE = float(nums[3])
                    values.append(f"{NICE / LP:.4f}")
                    break
                if "Average of disruptions:" in line:
                    extract = True

    output.append(",".join(values))

final_data = "\n".join(output)
with open("parsed.txt", "w+") as f:
    f.write(final_data)

