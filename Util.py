import numpy as np

def argmax(rng, a) : 
    """
    If there is a tie between multiple
    entries, we want to ensure that
    different entries are chosen.

    Parameters
    ----------
    rng : np.random.RandomState
        Random Number Generator
    a : np.ndarray
        1D array.
    """
    maxs = a == np.max(a)
    nTrue = np.sum(maxs)
    probs = maxs / nTrue
    return rng.choice(range(a.size), p=probs)

def argmin(rng, a) : 
    """
    If there is a tie between multiple
    entries, we want to ensure that
    different entries are chosen.

    Parameters
    ----------
    rng : np.RandomState
        Random Number Generator.
    a : np.ndarray
        1D Array.
    """
    mins = a == np.min(a)
    nTrue = np.sum(mins)
    probs = mins / nTrue
    return rng.choice(range(a.size), p=probs)

