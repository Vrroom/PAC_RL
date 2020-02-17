import numpy as np
import math
from MDP import *

def QSolver (mdp, P, Qinit, stoppingCondition) :
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
    stoppingCondition : lambda
        A function which takes the iteration
        count and difference between
        successive Q-functions and decides
        whether to stop or not.
    """
    iterCnt = 0
    error = math.inf
    Q = Qinit
    while not stoppingCondition(iterCnt, error) :
        Qold = np.copy(Q)
        V = np.max(Q, axis=1)
        Q = mdp.R + mdp.gamma * np.sum (P * V, axis=2)
        iterCnt += 1
        error = np.linalg.norm(Q - Qold)
    return Q

def occupancySolver (mdp, policy, P, stoppingCondition) :
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
    stoppingCondition : lambda
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

    P = P[np.arange(mdp.S), policy, :]

    while not stoppingCondition(iterCnt, error) :
        muOld = np.copy(mu)
        mu = Pstart + mdp.gamma * np.sum(P * mu, axis=1)
        iterCnt += 1
        error = np.linalg.norm(mu - muold)
    return mu

