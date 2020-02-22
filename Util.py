import numpy as np

RNG = np.random.RandomState(0)

def confidenceRadius (mdp, visitCount, delta) :
    """
    Referred to as omega in the DDV paper.
    Some magic function probably used to
    make the PAC guarantees go through.

    Parameters
    ----------
    mdp : MDP
        Underlying MDP.
    visitCount : int
        Number of visits to a particular
        (state, action) pair for which
        we are calculating the radius.
    delta : float
        A confidence interval parameter.
    """
    top = np.log(2 ** mdp.S - 2) - np.log(delta)
    return np.sqrt(2 * top / visitCount)


def argmax(a) : 
    """
    If there is a tie between multiple
    entries, we want to ensure that
    different entries are chosen.

    Parameters
    ----------
    a : np.ndarray
        1D array.
    """
    maxs = a == np.max(a)
    nTrue = np.sum(maxs)
    probs = maxs / nTrue
    return RNG.choice(range(a.size), p=probs)

def argmin(a) : 
    """
    If there is a tie between multiple
    entries, we want to ensure that
    different entries are chosen.

    Parameters
    ----------
    a : np.ndarray
        1D Array.
    """
    mins = a == np.min(a)
    nTrue = np.sum(mins)
    probs = mins / nTrue
    return RNG.choice(range(a.size), p=probs)

