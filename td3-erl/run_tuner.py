import wandb

# Command:
#  $wandb sweep sweep_config.yaml
# To generate the sweep remote manager
sweep_id = 'vgavra/sweeps-td3/1rh41mtb'

# Nubmer of runs doen by the lcoal agent:
count    =  15

# Start the local agent to work for the remote sweep:
wandb.agent(sweep_id = sweep_id, count=count)