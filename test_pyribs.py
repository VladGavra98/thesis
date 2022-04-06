import time
import matplotlib.pyplot as plt
import numpy as np


from ribs.optimizers import Optimizer
from ribs.archives import CVTArchive
from ribs.emitters import ImprovementEmitter
def simulate(solutions, link_lengths):
    """Returns the objective values and BCs for a batch of solutions.
    
    Args:
        solutions (np.ndarray): A (batch_size, dim) array where each row
            contains the joint angles for the arm. `dim` will always be 12
            in this tutorial.
        link_lengths (np.ndarray): A (dim,) array with the lengths of each
            arm link (this will always be an array of ones in the tutorial).
    Returns:
        objs (np.ndarray): (batch_size,) array with objective values.
        bcs (np.ndarray): (batch_size, 2) array with a BC in each row.
    """
    n_dim = link_lengths.shape[0]
    objs = -np.std(solutions, axis=1)

    # theta_1, theta_1 + theta_2, ...
    cum_theta = np.cumsum(solutions, axis=1)
    # l_1 * cos(theta_1), l_2 * cos(theta_1 + theta_2), ...
    x_pos = link_lengths[None] * np.cos(cum_theta)
    # l_1 * sin(theta_1), l_2 * sin(theta_1 + theta_2), ...
    y_pos = link_lengths[None] * np.sin(cum_theta)

    bcs = np.concatenate(
        (
            np.sum(x_pos, axis=1, keepdims=True),
            np.sum(y_pos, axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objs, bcs


dof = 12  # Degrees of freedom for the arm.
link_lengths = np.ones(dof)  # 12 links, each with length 1.
max_pos = np.sum(link_lengths)

archive = CVTArchive(
    # Number of bins.
    10000,
    # The x and y coordinates are bound by the maximum arm position.
    [(-max_pos, max_pos), (-max_pos, max_pos)],
    # The archive will use a k-D tree to search for the bin a solution
    # belongs to.
    use_kd_tree=True,
)


# emitters = [
#     ImprovementEmitter(
#         archive,
#         np.zeros(dof),
#         # Initial step size of 0.1 seems reasonable based on the bounds.
#         0.1,
#         bounds=[(-np.pi, np.pi)] * dof,
#         batch_size=30,
#     ) for _ in range(5)
# ]


# opt = Optimizer(archive, emitters)


# metrics = {
#     "Archive Size": {
#         "itrs": [0],
#         "vals": [0],  # Starts at 0.
#     },
#     "Max Objective": {
#         "itrs": [],
#         "vals": [],  # Does not start at 0.
#     },
# }

# start_time = time.time()
# total_itrs = 700
# for itr in range(1, total_itrs + 1):
#     sols = opt.ask()
#     objs, bcs = simulate(sols, link_lengths)
#     opt.tell(objs, bcs)

#     # Logging.
#     if itr % 50 == 0:
#         metrics["Archive Size"]["itrs"].append(itr)
#         metrics["Archive Size"]["vals"].append(len(archive))
#         metrics["Max Objective"]["itrs"].append(itr)
#         metrics["Max Objective"]["vals"].append(archive.stats.obj_max)
#         print(f"Finished {itr} itrs after {time.time() - start_time:.2f} s")