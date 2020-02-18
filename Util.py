import numpy as np

def stochasticArgmax(rng, sigma, a, axis=None) : 
    """
    If there is a tie between multiple
    entries, we want to reduce the probability 
    that the same entry is always chosen.

    Parameters
    ----------
    sigma : float
        Noise variance. Ideally should be 
        smaller than the errors we are willing
        to tolerate.
    a : np.ndarray
        Array.
    axis : int or sequence
        Axis along which to compute argmax.
    """
    noise = sigma * rng.randn(*a.shape)
    return np.argmax(a + noise, axis)

def stochasticArgmin(rng, sigma, a, axis=None) : 
    """
    If there is a tie between multiple
    entries, we want to reduce the probability 
    that the same entry is always chosen.

    Parameters
    ----------
    rng : np.RandomState
        Random Number Generator.
    sigma : float
        Noise variance. Ideally should be 
        smaller than the errors we are willing
        to tolerate.
    a : np.ndarray
        Array.
    axis : int or sequence
        Axis along which to compute argmin.
    """
    noise = sigma * rng.randn(*a.shape)
    return np.argmin(a + noise, axis)

