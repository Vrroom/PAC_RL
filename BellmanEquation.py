import numpy as np
from pulp import *
from itertools import product
import math
from MDP import *
from Util import *

def QSolver (mdp, P, Qinit, stop) :
    """
    Iterate the Bellman Optimality Equations
    to solve for the Q-function. 

    Parameters
    ----------
    mdp : MDP
        MDP object with rewards, discount factor
        and other relevant information.
    P : np.ndarray
        Estimates of the Transition Probabilities.
    Qinit : np.ndarray
        Initial estimates of Q-function.
    stop : lambda
        A function which takes the iteration
        count and difference between
        successive Q-functions and decides
        whether to stop or not.
    """
    iterCnt = 0
    error = math.inf
    Q = np.copy(Qinit)
    while not stop(iterCnt, error) :
        Qold = np.copy(Q)
        V = np.max(Q, axis=1)
        Q = mdp.R + mdp.gamma * np.sum (P * V, axis=2)
        iterCnt += 1
        error = np.linalg.norm(Q - Qold)
    return Q

def occupancySolver (mdp, policy, P, stop) :
    """
    Find the stationary occupancy of different 
    states under a given policy. 

    This is done by iterating the bellman 
    equations for the occupancy measure till
    convergence

    Parameters
    ----------
    mdp : MDP
        MDP object with rewards, discount factor
        and other relevant information.
    policy : np.ndarray
        Prescribed action for each state.
    P : np.ndarray
        Estimates of the Transition Probabilities.
    stop : lambda
        A function which takes the iteration
        count and difference between
        successive occupancy measures and decides
        whether to stop or not.
    """
    iterCnt = 0
    error = math.inf

    mu = np.zeros(mdp.S)

    Pstart = np.zeros(mdp.S)
    Pstart[mdp.s0] = 1

    P_ = np.copy(P)
    P_ = P_[np.arange(mdp.S), policy, :]

    while not stop(iterCnt, error) :
        muOld = np.copy(mu)
        mu = Pstart + mdp.gamma * np.sum(mu * P_.T, axis=1)
        iterCnt += 1
        error = np.linalg.norm(mu - muOld)
    return mu

def QBoundsSolver (mdp, PHat, Qinit, N, delta, sense, stop) :
    """
    Solve the Bellman Equations (7) and (8) 
    from the DDV paper:

    PAC Optimal MDP Planning with Application to 
    Invasive Species Management. (Taleghan et al. - 2013)

    Parameters
    ----------
    mdp : MDP
        Underlying MDP.
    PHat : np.ndarray
        Estimate of the MDP's 
        transition probabilities.
    Qinit : np.ndarray
        Initial guess of action-
        value function.
    N : np.ndarray
        State-Action visit count.    
    delta : float
        Confidence Interval Parameter
    sense : bool
        If true, then we solve
        to find the upper bound.
        Else the lower bound.
    stop : lambda
        The stopping condition.
    """
    
    def shiftP (Q, s, a, omega) :
        """
        Helper function used to do value
        iteration with confidence intervals.

        This function gives the probability 
        distribution in the confidence interval
        of the transition probability function
        that will maximize/minimize the outer 
        max/min in the Bellman Equation.

        Based on the procedure described in :

        An Analysis of model-based Interval Estimation
        for Markov Decision Processes. (Strehl, Littman - 2008)
        
        Parameters
        ----------
        Q : np.ndarray
            Action-Value function
        s : int
            State.
        a : int 
            Action.
        omega : float
            Confidence Interval width.
        """
        V = np.max(Q, axis=1)
        Pt = np.copy(PHat[s, a])

        addSelect = argmax if sense else argmin
        subSelect = argmin if sense else argmax
        
        val1 = -math.inf if sense else math.inf
        val2 = math.inf if sense else -math.inf

        # First add amount omega
        # to all the promising states.
        addAmount = omega
        while addAmount > 1e-5 : 
            V1 = np.copy(V)
            mask = Pt < 1
            V1[~mask] = val1
            s = addSelect(V1)
            zeta = min(1 - Pt[s], addAmount)
            Pt[s] += zeta
            addAmount -= zeta

        # Then subtract that value
        # from the less promising states.
        subAmount = omega
        while subAmount > 1e-5 :
            V1 = np.copy(V)
            mask = Pt > 0
            V1[~mask] = val2
            s = subSelect(V1)
            zeta = min(Pt[s], subAmount)
            Pt[s] -= zeta
            subAmount -= zeta

        return Pt / np.sum(Pt)

    iterCnt = 0
    error = math.inf
    Q = np.copy(Qinit)

    while not stop(iterCnt, error) :
        Qold = np.copy(Q)

        Pt = np.zeros(PHat.shape)
        for s, a in product(range(mdp.S), range(mdp.A)) :
            omega = confidenceRadius(mdp, N[s, a], delta) / 2
            Pt[s, a] = shiftP(Q, s, a, omega)

        Q = QSolver(mdp, Pt, Q, stop)

        iterCnt += 1
        error = np.linalg.norm(Q - Qold)
    return Q

def QPiBoundsSolver (mdp, PHat, Qinit, N, delta, sense, stop, pi) :
    """
    Solve the Bellman Equations (7) and (8) 
    from the DDV paper:

    PAC Optimal MDP Planning with Application to 
    Invasive Species Management. (Taleghan et al. - 2013)

    Parameters
    ----------
    mdp : MDP
        Underlying MDP.
    PHat : np.ndarray
        Estimate of the MDP's 
        transition probabilities.
    Qinit : np.ndarray
        Initial guess of action-
        value function.
    N : np.ndarray
        State-Action visit count.    
    delta : float
        Confidence Interval Parameter
    sense : bool
        If true, then we solve
        to find the upper bound.
        Else the lower bound.
    stop : lambda
        The stopping condition.
    """
    
    def shiftP (Q, s, a, omega) :
        """
        Helper function used to do value
        iteration with confidence intervals.

        This function gives the probability 
        distribution in the confidence interval
        of the transition probability function
        that will maximize/minimize the outer 
        max/min in the Bellman Equation.

        Based on the procedure described in :

        An Analysis of model-based Interval Estimation
        for Markov Decision Processes. (Strehl, Littman - 2008)
        
        Parameters
        ----------
        Q : np.ndarray
            Action-Value function
        s : int
            State.
        a : int 
            Action.
        omega : float
            Confidence Interval width.
        """
        V = Q[np.arange(mdp.S), pi]
        Pt = np.copy(PHat[s, a])

        addSelect = argmax if sense else argmin
        subSelect = argmin if sense else argmax
        
        val1 = -math.inf if sense else math.inf
        val2 = math.inf if sense else -math.inf

        # First add amount omega
        # to all the promising states.
        addAmount = omega
        while addAmount > 1e-5 : 
            V1 = np.copy(V)
            mask = Pt < 1
            V1[~mask] = val1
            s = addSelect(V1)
            zeta = min(1 - Pt[s], addAmount)
            Pt[s] += zeta
            addAmount -= zeta

        # Then subtract that value
        # from the less promising states.
        subAmount = omega
        while subAmount > 1e-5 :
            V1 = np.copy(V)
            mask = Pt > 0
            V1[~mask] = val2
            s = subSelect(V1)
            zeta = min(Pt[s], subAmount)
            Pt[s] -= zeta
            subAmount -= zeta

        return Pt / np.sum(Pt)

    iterCnt = 0
    error = math.inf
    Q = np.copy(Qinit)

    while not stop(iterCnt, error) :
        Qold = np.copy(Q)

        Pt = np.zeros(PHat.shape)
        for s, a in product(range(mdp.S), range(mdp.A)) :
            omega = confidenceRadius(mdp, N[s, a], delta) / 2
            Pt[s, a] = shiftP(Q, s, a, omega)

        Q = QSolver(mdp, Pt, Q, stop)

        iterCnt += 1
        error = np.linalg.norm(Q - Qold)
    return Q
