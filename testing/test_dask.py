import time
import random
from timeit import timeit
import pandas as pd
import numpy as np
import dask
from dask.distributed import Client, progress


def costly_simulation(list_param):
    time.sleep(random.random())
    return sum(list_param)


if __name__ == '__main__':
    np.random.seed(7)
    input_params = pd.DataFrame(np.random.random(size=(500, 4)),
                                columns=['param_a', 'param_b', 'param_c', 'param_d'])


    results,lazy_results = [], []

    # standard computation
    t0 = time.time()
    for parameters in input_params.values[:10]:
        result = costly_simulation(parameters)
        results.append(result)
    t1 = time.time()

    print(results[0])
    print(f'Normal time: {(t1 - t0) *10**3:.03f}ms')

    # init dask client obj
    client = Client(threads_per_worker=20, n_workers=7)
    client.cluster.scale(10)  # ask for ten 4-thread workers

    # parallel computation
    t0 = time.time()
    for parameters in input_params.values[:10]:
        lazy_result = dask.delayed(costly_simulation)(parameters)
        lazy_results.append(lazy_result)

    futures = dask.persist(*lazy_results) 
    results = dask.compute(*futures)

    t2= time.time()
    print(results[0])
    print(f'Lazy results time: {(t2 - t0)*10**3:.03f}ms')

    #vizualize
    lazy_result.vizualize()