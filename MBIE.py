import numpy as np
from MDP import *
from BellmanEquation import *
from itertools import product
from Util import *

class MBIE () :
    """
    MBIE-reset algorithm.
    """

    def __init__ (self, mdp, epsilon, delta, seed) :
        """
        Constructor.

        Parameters
        ----------
        mdp : MDP
            MDP to solve.
        epsilon : float
            PAC Accuracy parameter.
        delta : float
            PAC Confidence parameter.
        seed : int
            Seed for the random number
            generator. To ensure 
            reproducibility.
        """
        self.mdp = mdp

        self.epsilon = epsilon
        self.delta = delta

        self.N = np.zeros(mdp.T.shape)
        self.Ntotal = np.zeros(mdp.R.shape)
        self.PHat = np.ones(mdp.T.shape)
        self.omega =  np.zeros(mdp.R.shape)

        self.m = self.numberOfSamples()
        self.delta_ = self.delta / (mdp.S * mdp.A * self.m)

        self.QUpper = np.ones(mdp.R.shape) * mdp.Vmax

        self.mu = np.zeros(mdp.S)

        # Since rewards are deterministic
        # this variable helps us keep track
        # of them once we have encountered
        # a (s, a) pair.
        self.R = np.zeros(mdp.R.shape)

        self.rng = np.random.RandomState(seed)

        self.H = 10

        # In a lot of cases, we have 
        # to solve the bellman equations
        # iteratively. This is the stopping
        # predicate.
        self.stop = lambda i, err : err < 1e-9

        for _ in range(1):
            self.uniformSample()

        self.mbieLoop()

    def uniformSample (self) :
        """
        Uniformly sample all (s, a) once
        to get an initial estimate of 
        transition probabilities.
        """
        S = self.mdp.S
        A = self.mdp.A

        for s, a in product(range(S), range(A)):
            s_, self.R[s, a] = self.mdp.step(s, a)
            self.updateVisitStatistics(s, a, s_)

    def mbieLoop (self) :
        """
        Main loop of the DDV algorithm with 
        the OOU Heuristic to compute the 
        occupancy measure.
        """
        self.iterCnt = 0
        while self.iterCnt < 5000:
            s = self.mdp.s0
            for h in range(self.H) :
                self.QUpper = QBoundsSolver(self.mdp, self.PHat, self.QUpper, self.Ntotal, 0.1, True, self.stop)
                a = np.argmax(self.QUpper[s])
                s_, self.R[s,a] = self.mdp.step(s, a)
                self.updateVisitStatistics(s, a, s_)
                s = s_

            if self.iterCnt % 10 == 0: 
                print(self.iterCnt)
                print(self.QUpper)

            self.iterCnt += 1

    def numberOfSamples (self) :
        """
        The number of times you have to sample
        a particular (state, action) so that the
        uncertainty in its Q-function is small.
        """
        S = self.mdp.S
        A = self.mdp.A
        gamma = self.mdp.gamma

        factor = 1 / (self.epsilon ** 2 * (1 - gamma) ** 4)
        term2 = np.log((S * A) / (self.epsilon * (1 - gamma) ** self.delta))
        return (S + term2) * factor
        
    def updateVisitStatistics(self, s, a, s_) :
        """
        Update how many times we have taken
        action a from state s, how many
        times we have reached s_ by doing so
        and what is the new estimate of transition
        probabilities as a result and the confidence
        interval which changes due to more visits.
        
        Parameters
        ----------
        s : int
            State.
        a : int
            Action.
        """
        self.N[s, a, s_] += 1
        self.Ntotal[s, a] += 1
        self.PHat[s, a] = self.N[s, a] / self.Ntotal[s, a]
        self.omega[s, a] = confidenceRadius(self.mdp, self.Ntotal[s, a], self.delta_)
