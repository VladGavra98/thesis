#!/bin/bash
python3.8 ./tuner.py  --batch_size=128   --hidden_size=72 --lr=0.0009812 --noise_sd=0.313487 --run_name='crimson' --use_caps=True --should_log --next_save=200
python3.8 ./tuner.py  --batch_size=128   --hidden_size=72 --lr=0.0009812 --noise_sd=0.3134872 --run_name='crimson_nocaps' --use_caps=False --should_log --next_save=200
python3.8 ./tuner.py  --batch_size=64   --hidden_size=96 --lr=0.0004335 --noise_sd=0.3363253 --run_name='zesty' --use_caps=True --should_log --next_save=200
python3.8 ./tuner.py  --batch_size=64  --hidden_size=96 --lr=0.0004335 --noise_sd=0.3363253 --run_name='zesty_nocaps' --use_caps=False --should_log --next_save=200