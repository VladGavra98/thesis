import wandb

sweep_id, count = "vgavra/sweeps-td3/gskafvym", 20
wandb.agent(sweep_id = sweep_id, count=count)