import numpy as np

def HR2T(sample): # Hermans-Rasson 2 T test
    d = sample[None, :] - sample[:, None]
    T = np.sum(np.abs(np.abs(d) - np.pi) - (np.pi / 2) - 2.895 * (np.abs(np.sin(d)) - (2 / np.pi))) / sample.size
    return T

def HR2P(sample, univals = 1000, seed=None): # Hermans-Rasson 2 p-value
    rng = np.random.default_rng(seed=seed)
    Tsample = HR2T(sample)
    n = sample.size
    testset = np.zeros(univals)
    for f in range(univals):
        data1 = rng.uniform(size=n, low=0, high=2*np.pi)
        testset[f] = HR2T(data1)
    p = (np.sum(testset > Tsample) + 1) / (univals + 1)
    return p