Naming convention guide: `s{a}_d{b}_m{c}_b{d}.ckpt` means that: 

- Seed `a` was used to train the model
- The model was trained with a scheduling density of `b`
- If `c` is 1 and `d` is 0, that means we used the move-up crew reward function to train the model. Note that we did not include these experiments in the paper because, even with an integer program that optimally inserts move-up crews into a schedule, the schedule barely gains any resistance to disruption.
- If `c` is 0 and `d` is 1, that means we used the buffer reward function to train the model.