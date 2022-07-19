import wandb


sweep_id, count = 'vgavra/sweeps-td3/rd2o46sf', 15
wandb.agent(sweep_id = sweep_id, count=count)