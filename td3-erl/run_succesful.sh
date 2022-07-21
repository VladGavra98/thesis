#!/bin/bash
python3.8 ./tuner.py  --batch_size=128   --hidden_size=72 --lr=0.0009812 --noise_sd=0.3134872  --should_log  --run_name="crimson_nocaps" --next_save=200
python3.8 ./tuner.py  --batch_size=64  --hidden_size=96 --lr=0.0004335 --noise_sd=0.3363253   --should_log --run_name="zesty_nocaps" --next_save=200