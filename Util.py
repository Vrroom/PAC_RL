import numpy as np
import mdptoolbox.example
from itertools import chain, combinations

RNG = np.random.RandomState(20)

def powerset(iterable):
    """
    Stolen from :

    https://docs.python.org/3/library/itertools.html

    Helps in enumerating all two action
    possible policies.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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

def log (thing, types) :
    """
    Logs all member variables 
    of a certain type from the object.
    """
    members = dir(thing)
    print("-----------")
    for m in members : 
        attr = getattr(thing, m) 
        if any([isinstance(attr, t) for t in types]) : 
            print(m)
            print(attr)
    print("-----------")

