import numpy as np
from tqdm import tqdm
import numba

# @numba.jit()
# def HR2T(sample):
#     n = sample.size
#     total = 0
#     for i in range(n):
#         for j in range(n):
#             total = total + abs(abs(sample[i]-sample[j])-np.pi)-(np.pi/2)
#             total = total - (2.895*(abs(np.sin(sample[i]-sample[j]))-(2/np.pi)))
#     T = total/n
#     return T


@numba.jit(parallel=True, nopython = False)
def HR2T(sample):
    n = sample.size
    # total = 0
    local_sums = np.zeros(n)
    for i in numba.prange(n):
        local_sum = 0
        for j in range(n):
            local_sum = local_sum + abs(abs(sample[i]-sample[j])-np.pi)-(np.pi/2)
            local_sum = local_sum - (2.895*(abs(np.sin(sample[i]-sample[j]))-(2/np.pi)))
        local_sums[i] = local_sum
    
    total = 0
    for i in range(n):
        total += local_sums[i]
        
    T = total/n
    return T


# def HR2T(sample): # Hermans-Rasson 2 T test
#     d = sample[None, :] - sample[:, None]
#     T = np.sum(np.abs(np.abs(d) - np.pi) - (np.pi / 2) - 2.895 * (np.abs(np.sin(d)) - (2 / np.pi))) / sample.size
#     return T

def HR2P(sample, univals = 1000, seed=None, progress_bar = False): # Hermans-Rasson 2 p-value
    rng = np.random.default_rng(seed=seed)
    Tsample = HR2T(sample)
    n = sample.size
    testset = np.zeros(univals)
    if progress_bar:
        loop = tqdm(range(univals))
    else:
        loop = range(univals)
    for f in loop:
        data1 = rng.uniform(size=n, low=0, high=2*np.pi)
        testset[f] = HR2T(data1)
    p = (np.sum(testset > Tsample) + 1) / (univals + 1)
    # p = (np.sum(testset > Tsample)) / (univals + 1)
    return p