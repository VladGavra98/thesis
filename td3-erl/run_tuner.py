import wandb


sweep_id, count = 'vgavra/sweeps-td3/w2a09vih', 10
wandb.agent(sweep_id = sweep_id, count=count)