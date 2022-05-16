"""Uses CMA-ME to train linear agents in Lunar Lander.

This script uses the same setup as the tutorial, but it also uses Dask to
parallelize evaluations on a single machine and adds in a CLI. Refer to the
tutorial here: https://docs.pyribs.org/en/stable/tutorials/lunar_lander.html for
more info.

You should not need much familiarity with Dask to read this example. However, if
you would like to know more about Dask, we recommend referring to the quickstart
for Dask distributed: https://distributed.dask.org/en/latest/quickstart.html.

This script creates an output directory (defaults to `lunar_lander_output/`, see
the --outdir flag) with the following files:

    - archive.csv: The CSV representation of the final archive, obtained with
      as_pandas().
    - archive_ccdf.png: A plot showing the (unnormalized) complementary
      cumulative distribution function of objective values in the archive. For
      each objective value p on the x-axis, this plot shows the number of
      solutions that had an objective value of at least p.
    - heatmap.png: A heatmap showing the performance of solutions in the
      archive.
    - metrics.json: Metrics about the run, saved as a mapping from the metric
      name to a list of x values (iteration number) and a list of y values
      (metric value) for that metric.
    - {metric_name}.png: Plots of the metrics, currently just `archive_size` and
      `max_score`.

In evaluation mode (--run-eval flag), the script will read in the archive from
the output directory and simulate 10 random solutions from the archive. It will
write videos of these simulations to a `videos/` subdirectory in the output
directory.

Usage:
    # Basic usage - should take ~1 hour with 4 cores.
    python lunar_lander.py NUM_WORKERS
    # Now open the Dask dashboard at http://localhost:8787 to view worker
    # status.

    # Evaluation mode. If you passed a different outdir and/or env_seed when
    # running the algorithm with the command above, you must pass the same
    # outdir and/or env_seed here.
    python lunar_lander.py --run-eval
Help:
    python lunar_lander.py --help
"""
# basic OS interaction
import time
from pathlib import Path

# python standard modules for AI
import gym
import numpy as np
import pandas as pd
from tqdm import tqdm
from dask.distributed import Client, LocalCluster

# pyribs
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter, RandomDirectionEmitter
from ribs.optimizers import Optimizer

# my modules
from QD.saving_utils import *
from envs.lunarlander import simulate


env_name = "LunarLanderContinuous-v2"


class QD_agent:

    def __init__(self, model, env):
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        self.model = model.reshape((action_dim, obs_dim))

    def select_action(self, state):
        return self.model @ state


def create_optimizer(seed, n_emitters, sigma0, batch_size):
    """Creates the Optimizer based on given configurations.

    See lunar_lander_main() for description of args.

    Returns:
        A pyribs optimizer set up for CMA-ME (i.e. it has ImprovementEmitter's
        and a GridArchive).
    """
    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    print(f'Action-sapce dim: {action_dim} \nObservation space dim: {obs_dim}')
    archive = GridArchive(
        [50, 50],  # 50 bins in each dimension.
        [(-1.0, 1.0), (-3.0, 0.0)],  # (-1, 1) for x-pos and (-3, 0) for y-vel.
        seed=seed,
    )

    # If we create the emitters with identical seeds, they will all output the
    # same initial solutions. The algorithm should still work -- eventually, the
    # emitters will produce different solutions because they get different
    # responses when inserting into the archive. However, using different seeds
    # avoids this problem altogether.
    seeds = ([None] * n_emitters
             if seed is None else [seed + i for i in range(n_emitters)])
    initial_model = np.zeros((action_dim, obs_dim))
    emitters = [
        ImprovementEmitter(
            archive,
            initial_model.flatten(),
            sigma0=sigma0,
            batch_size=batch_size,
            seed=s,
        ) for s in seeds
    ]

    optimizer = Optimizer(archive, emitters)
    return optimizer


def run_search(client, optimizer, env_seed, iterations, log_freq):
    """Runs the QD algorithm for the given number of iterations.

    Args:
        client (Client): A Dask client providing access to workers.
        optimizer (Optimizer): pyribs optimizer.
        env_seed (int): Seed for the environment.
        iterations (int): Iterations to run.
        log_freq (int): Number of iterations to wait before recording metrics.
    Returns:
        dict: A mapping from various metric names to a list of "x" and "y"
        values where x is the iteration and y is the value of the metric. Think
        of each entry as the x's and y's for a matplotlib plot.
    """
    print(
        "> Starting search.\n"
        "  - Open Dask's dashboard at http://localhost:8787 to monitor workers."
    )

    metrics = {
        "Max Score": {
            "x": [],
            "y": [],
        },
        "Archive Size": {
            "x": [0],
            "y": [0],
        },
    }

    start_time = time.time()

    for itr in tqdm(range(1, iterations + 1)):
        # Request models from the optimizer.
        sols = optimizer.ask()

        # Evaluate the models and record the objectives and BCs.
        objs, bcs = [], []

        # Ask the Dask client to distribute the simulations among the Dask
        # workers, then gather the results of the simulations.
        futures = client.map(lambda model: simulate(
            model, env_name, env_seed), sols)
        results = client.gather(futures)

        # Process the results.
        for obj, impact_x_pos, impact_y_vel in results:
            objs.append(obj)
            bcs.append([impact_x_pos, impact_y_vel])

        # Send the results back to the optimizer.
        optimizer.tell(objs, bcs)

        # Logging.
        # progress()
        if itr % log_freq == 0 or itr == iterations:
            elapsed_time = time.time() - start_time
            metrics["Max Score"]["x"].append(itr)
            metrics["Max Score"]["y"].append(
                optimizer.archive.stats.obj_max)
            metrics["Archive Size"]["x"].append(itr)
            metrics["Archive Size"]["y"].append(len(optimizer.archive))
            print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
            print(f"  - Max Score: {metrics['Max Score']['y'][-1]}")
            print(f"  - Archive Size: {metrics['Archive Size']['y'][-1]}")

    return metrics


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                              Test Traiend Policies
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_evaluation(outdir, env_seed=1339, random: bool = False, broken_engine: bool = False, trials: bool = 5):
    """Simulates 10 random archive solutions and saves videos of them.

    Videos are saved to outdir / videos.

    Args:
        outdir (Path): Path object for the output directory from which to
            retrieve the archive and save videos.
        env_seed (int): Seed for the environment.
    """
    outdir = Path(outdir)
    df = pd.read_csv(outdir / "archive.csv")

    if random:
        indices = np.random.permutation(len(df))[:10]
    else:
        indices = range(len(df))

    # if do_plot:
    #     save_heatmap(df, str(outdir / "heatmap.png"))
    # Use a single env so that all the videos go to the same directory.
    if broken_engine:
        print('Environment: ' + env_name + ' broken engine')
    else:
        print('Environment: ' + env_name)

    # Since we are using multiple processes, it is simpler if each worker
    # just creates their own copy of the environment instead of trying to
    # share the environment. This also makes the function "pure."
    env = gym.make(env_name)

    video_env = gym.wrappers.Monitor(
        gym.make(env_name),
        str(outdir / "videos"),
        force=True,
        # Default is to write the video for "cubic" episodes -- 0,1,8,etc (see
        # https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py#L54).
        # This will ensure all the videos are written.
        video_callable=lambda idx: True,
    )

    if env_seed is not None:
        env.seed(env_seed)

    print(f'Evaluate archive:')
    MAX_REWARD = 200
    max_idx = 0
    for idx in tqdm(indices):
        model = np.array(df.loc[idx, "solution_0":])
        actor = QD_agent(model, env)   # wrap the model in an agent class
        reward, impact_x_pos, impact_y_vel = simulate(
            actor, env, broken_engine=broken_engine, render=False)

        if reward > MAX_REWARD:
            print(f"Max reward {reward:0.3} at index = {idx}")
            elite_actor = actor
            MAX_REWARD = reward
            max_idx = idx

    # simulate again the best, this time with video
    print(f'\nEvaluating elite (agent {max_idx})...')
    rewards, bcs = [], []
    for _ in tqdm(range(trials)):
        reward, impact_x_pos, impact_y_vel = simulate(
            elite_actor, env, broken_engine=broken_engine, render=False)
        rewards.append(reward)
        bcs.append((impact_x_pos, impact_y_vel))

    bcs = np.asarray(bcs)

    simulate(elite_actor, video_env, broken_engine=broken_engine, render=True)
    # close video env
    video_env.close()

    return np.average(rewards), np.std(rewards), np.average(bcs, axis=0)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                    Main
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def lunar_lander_main(workers=4,
                      env_seed=1339,
                      iterations=400,
                      log_freq=25,
                      n_emitters=5,
                      batch_size=25,
                      sigma0=1.0,
                      seed=None,
                      outdir="lunar_lander_output"):
    """Uses CMA-ME to train linear agents in Lunar Lander.

    Args:
        workers (int): Number of workers to use for simulations.
        env_seed (int): Environment seed. The default gives the flat terrain
            from the tutorial.
        iterations (int): Number of iterations to run the algorithm.
        log_freq (int): Number of iterations to wait before recording metrics
            and saving heatmap.
        n_emitters (int): Number of emitters.
        batch_size (int): Batch size of each emitter.
        sigma0 (float): Initial step size of each emitter.
        seed (seed): Random seed for the pyribs components.
        outdir (str): Directory for Lunar Lander output.
        run_eval (bool): Pass this flag to run an evaluation of 10 random
            solutions selected from the archive in the `outdir`.
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    # Setup Dask. The client connects to a "cluster" running on this machine.
    # The cluster simply manages several concurrent worker processes. If using
    # Dask across many workers, we would set up a more complicated cluster and
    # connect the client to it.
    print(
        f"Total nubmer of simualtions: {iterations * batch_size * n_emitters}")
    cluster = LocalCluster(
        processes=True,  # Each worker is a process.
        n_workers=workers,  # Create this many worker processes.
        threads_per_worker=1,  # Each worker process is single-threaded.
    )
    client = Client(cluster)

    # CMA-ME
    optimizer = create_optimizer(seed, n_emitters, sigma0, batch_size)
    metrics = run_search(client, optimizer, env_seed, iterations, log_freq)

    # Outputs
    optimizer.archive.as_pandas().to_csv(outdir / "archive.csv")
    save_ccdf(optimizer.archive, str(outdir / "archive_ccdf.png"))
    save_heatmap(optimizer.archive, str(outdir / "heatmap.png"))
    save_metrics(outdir, metrics)

    return 0


if __name__ == "__main__":
    # Declare folder for results
    outdir_path = './Results_QD/run6_continuous_balanced'

    # Train
    # lunar_lander_main(outdir=outdir_path, workers = 5)

    # Evaluate
    reward_mean, reward_std, bcs = run_evaluation(
        outdir=outdir_path, trials=100, broken_engine=False)
    print(f"Reward: {reward_mean:.2f}\n",
          f"Reward STD: {reward_std:.2f}\n",
          f"Impact x-pos: {bcs[0]:.2f}\n",
          f"Impact y-vel: {bcs[1]:.2f}\n")