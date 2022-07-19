import wandb


sweep_id, count = 'vgavra/sweeps-td3/qgqt6qbw', 12
wandb.agent(sweep_id = sweep_id, count=count)