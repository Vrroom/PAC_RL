import numpy as np
from MDP import *
from BellmanEquation import *
from itertools import product
from Util import *

class DDV () :
    """
    The DDV algorithm uses an efficient
    sampling strategy to quickly find a
    good policy. 
    
    We give the entire MDP
    object to this algorithm but the 
    algorithm doesn't make use of the 
    MDP's underlying transition 
    probabilities or reward functions.

    Instead, it maintains its own estimates
    which are used by the sampling process.
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

        self.QUpper = np.ones(mdp.R.shape) * mdp.Vmax
        self.QLower = np.zeros(mdp.R.shape)

        self.mu = np.zeros(mdp.S)

        # Since rewards are deterministic
        # this variable helps us keep track
        # of them once we have encountered
        # a (s, a) pair.
        self.R = np.zeros(mdp.R.shape)

        self.rng = np.random.RandomState(seed)

        # In a lot of cases, we have 
        # to solve the bellman equations
        # iteratively. This is the stopping
        # predicate.
        self.stop = lambda i, err : i > 1000 or err < 1e-5

        for _ in range(100) :
            self.uniformSample()

        self.ddvLoop()

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
            self.updateVisitCountAndPHat(s, a, s_)

    def ddvLoop (self) :
        """
        Main loop of the DDV algorithm with 
        the OOU Heuristic to compute the 
        occupancy measure.
        """
        m = self.numberOfSamples()
        delta_ = self.delta / (self.mdp.S * self.mdp.A * m)
        
        exploredStates = np.zeros(self.mdp.S, dtype=bool)
        exploredStates[self.mdp.s0] = True

        self.ddv = np.zeros(self.mdp.R.shape)

        iterCnt = 0
        while True and iterCnt < 1000:
            self.updateQConfidenceIntervals(delta_)

            VUpper = np.max(self.QUpper[self.mdp.s0])
            VLower = np.max(self.QLower[self.mdp.s0])

            print(VUpper - VLower)
            if VUpper - VLower <= self.epsilon :
                break

            self.updateStationaryDistribution()

            for s in range(self.mdp.S) :
                if exploredStates[s] :
                    self.ddv[s] = np.array([self.computeDDV(s, a, delta_) for a in range(self.mdp.A)])
            
            s, a = np.unravel_index(argmax(self.ddv.flatten()), self.ddv.shape)
            s_, self.R[s,a] = self.mdp.step(s, a)

            print(s, a, s_)

            exploredStates[s_] = True

            self.updateVisitCountAndPHat(s, a, s_)

            iterCnt += 1

    def computeDDV(self, s, a, delta) :
        """
        The OOU heuristic is used to calculate
        an approximation of DDV. This is done
        by multiplying the stationary distribution
        of states under the current most optimistic
        policy and the change in the confidence interval
        of Q value for (s, a)

        Parameters
        ----------
        s : int
            State.
        a : int
            Action.
        delta : float
            Confidence measure.
        """
        dQ = self.QUpper[s, a] - self.QLower[s, a]
        if self.Ntotal[s, a] == 0 : 
            dQ_ = self.mdp.gamma * self.mdp.Vmax
            ddQ = np.abs(dQ - dQ_)
            return self.mu[s] * ddQ
        else :
            Pu = np.copy(self.PHat)
            Pl = np.copy(self.PHat)

            self.tryExploring(s, a)

            QUpper_ = QBoundsSolver(self.mdp, self.PHat, self.QUpper, self.Ntotal, delta, True, self.stop)
            QLower_ = QBoundsSolver(self.mdp, self.PHat, self.QLower, self.Ntotal, delta, False, self.stop)

            self.undoExploring(s, a)

            dQ_ = QUpper_[s, a] - QLower_[s, a]
            ddQ = np.abs(dQ - dQ_)
            return self.mu[s] * ddQ

    def tryExploring (self, s, a) :
        """
        Increase the visits to (s, a).

        We don't actually sample (s, a) because
        we would just like to know how the 
        Q(s, a) confidence interval will change.

        A call to this function must be followed
        by a call to undoExploring.

        Parameters
        ----------
        s : int
            State.
        a : int
            Action.
        """
        self.Ntotal[s, a] += 1

    def undoExploring (self, s, a) :
        """
        Undo the visits to (s, a)

        Follows the call to tryExploring.

        Parameters
        ----------
        s : int
            State.
        a : int
            Action.
        """
        self.Ntotal[s, a] -= 1

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

    def updateStationaryDistribution (self) :
        """
        After each change to our estimate
        of upper bound on Q-function and 
        the transition probabilities, we need
        to recompute what will be the new 
        stationary distribution of states
        under the most optimistic policy.
        """
        policy = np.argmax(self.QUpper, axis=1)
        self.mu = occupancySolver(self.mdp, policy, self.PHat, self.stop)
        
    def updateVisitCountAndPHat(self, s, a, s_) :
        """
        Update how many times we have taken
        action a from state s, how many
        times we have reached s_ by doing so
        and what is the new estimate of transition
        probabilities as a result.
        
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

    def updateQConfidenceIntervals(self, delta) :
        """
        Update Q-function confidence intervals. 
        Used in the ddvLoop function.
        
        The bounds are computed by first 
        finding the transition probabilities 
        which would maximize the bounds and
        then solving the bellman equations.

        This function is written separately
        to avoid clutter. 

        Parameters
        ----------
        delta : float
            Confidence Parameter for shifting
            probability mass.
        """
        self.QUpper = QBoundsSolver(self.mdp, self.PHat, self.QUpper, self.Ntotal, delta, True, self.stop)
        self.QLower = QBoundsSolver(self.mdp, self.PHat, self.QLower, self.Ntotal, delta, False, self.stop)
