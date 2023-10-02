import numpy as np

def HermansRasson2T(sample):
    d = sample[None, :] - sample[:, None]
    T = np.sum(np.abs(np.abs(d) - np.pi) - (np.pi / 2) - 2.895 * (np.abs(np.sin(d)) - (2 / np.pi))) / sample.size
    return T

def HermansRasson2P(sample, univals = 1000, seed=None):
    rng = np.random.default_rng(seed=seed)
    Tsample = HermansRasson2T(sample)
    n = sample.size
    testset = np.zeros(univals)
    for f in range(univals):
        data1 = rng.uniform(size=n, low=0, high=2*np.pi)
        testset[f] = HermansRasson2T(data1)
    p = (np.sum(testset > Tsample) + 1) / (univals + 1)
    return p