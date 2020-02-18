import numpy as np
from MDP import *
from BellmanEquation import *
from itertools import product

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

    def __init__ (self, mdp, epsilon, delta) :
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
        """
        self.mdp = mdp

        self.epsilon = epsilon
        self.delta = delta

        self.N = np.zeros(mdp.T.shape)
        self.Ntotal = np.zeros(mdp.R.shape)

        self.QUpper = np.ones(mdp.R.shape) * mdp.Vmax
        self.QLower = np.zeros(mdp.R.shape)

        self.mu = np.zeros(mdp.S)

        self.PHat = 0.5 * np.ones(mdp.T.shape)

        # Since rewards are deterministic
        # this variable helps us keep track
        # of them once we have encountered
        # a (s, a) pair.
        self.R = np.zeros(mdp.R.shape)

        # In a lot of cases, we have 
        # to solve the bellman equations
        # iteratively. This is the stopping
        # predicate.
        self.stop = lambda i, err : i > 100 or err < 0.01

        self.ddvLoop()

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

        while True :
            self.updateQConfidenceIntervals(delta_)

            VUpper = np.max(self.QUpper[self.mdp.s0])
            VLower = np.max(self.QLower[self.mdp.s0])

            if VUpper - VLower <= self.epsilon :
                break

            self.updateStationaryDistribution()

            for s in range(self.mdp.S) :
                if exploredStates[s] :
                    ddv[s] = np.array([self.computeDDV(s, a) for a in range(self.mdp.A)])
            
            s, a = np.unravel_index(ddv.argmax(), ddv.shape)
            s_, self.R[s,a] = self.mdp.step(s, a)

            exploredStates[s_] = True

            self.updateVisitCountAndPHat(s, a, s_)

    def computeDDV(self, s, a) :
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
        """
        dQ = self.QUpper[s, a] - self.QLower[s, a]
        if self.Ntotal[s, a] == 0 : 
            dQ_ = self.mdp.gamma * self.mdp.Vmax
            ddQ = np.abs(dQ - dQ_)
            return self.mu(s) * ddQ
        else :
            Pu = np.copy(self.PHat)
            Pl = np.copy(self.PHat)

            self.tryExploring(s, a)

            # TODO Not sure which delta this is?
            Pu[s, a] = self.shiftProbabilityMass(s, a, delta, True)
            Pl[s, a] = self.shiftProbabilityMass(s, a, delta, False)

            self.undoExploring(s, a)

            QUpper_ = QSolver(self.mdp, Pu, self.QUpper, self.stop)
            QLower_ = QSolver(self.mdp, Pu, self.QLower, self.stop)

            dQ_ = QUpper_[s, a] - QLower_[s, a]
            ddQ = np.abs(dQ - dQ_)
            return self.mu(s) * ddQ

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

    def shiftProbabilityMass (self, s, a, delta, findUpper) :
        """
        The algorithm searches for extremal bounds
        for the Q-function. It searches for these 
        extremal bounds within a particular 
        confidence interval of the transition
        probability estimates. 

        findUpper specifies how to shift the
        probability mass. When you want to find 
        an upper bound on the Q-function, you shift
        mass from less valuable states to more 
        valuable states. For the lower bound, you
        do the opposite.

        This algorithm by Strehl and Littman returns
        the transition probabilities. Plugging them
        in the Bellman Equation gives the extremal
        bound on the Q function.

        Parameters
        ----------
        s : int
            The state for which we are finding
            the transition probabilities that 
            maximise the Q function.
        a : int
            The action for which we are finding 
            the transition probabilities.
        delta : float
            Confidence Parameter.
        findUpper : bool 
            True if we have to find the upper
            bound, else False.
        """
        Pt = self.PHat
        
        if findUpper : 
            V = np.max(self.QUpper, axis=1)
        else : 
            V = np.max(self.QLower, axis=1)

        deltaOmega = self.confidenceRadius(s, a, delta) / 2

        while deltaOmega > 0 : 
            S_ = self.PHat[s, a] < 1

            donor = np.argmin(V[Pt[s, a] > 0])
            recipient = np.argmax(V[Pt[s, a] < 1 and S_])

            zeta = min(1 - Pt[s, a, donor], Pt[s, a, recipient], deltaOmega)

            if not findUpper :
                donor, recipient = recipient, donor

            Pt[s, a, donor] -= zeta
            Pt[s, a, recipient] += zeta 

            deltaOmega -= zeta

        return Pt[s, a]

    def shiftProbabilityMassGT (self, s, a, delta, M0, findUpper) :
        """
        The purpose of this function is the
        same as that of shiftProbabilityMass.

        There are some extensions which 
        are called Good-Turing Extensions
        which were proposed by the authors
        and are implemented here.

        Parameters
        ----------
        s : int
            The state for which we are finding
            the transition probabilities that 
            maximise the Q function.
        a : int
            The action for which we are finding 
            the transition probabilities.
        delta : float
            Confidence Parameter.
        M0 : float
            Missing Mass Limit. Don't 
            know what this actually means.
        findUpper : bool
            Whether to find P to upper bound
            Q or lower bound it.
        """
        Pt = self.PHat
        
        if findUpper : 
            V = np.max(self.QUpper, axis=1)
        else : 
            V = np.max(self.QLower, axis=1)

        constant1 = self.confidenceRadius(s, a, delta / 2) / 2
        # TODO : Find out what this is and wrap this up in a function.
        constant2 = (1 + 2**0.5) * (np.log(2 / delta) / saCount)**0.5
        deltaOmega = min(constant1, constant2)

        unvisitedSucc = self.N[s, a] == 0

        while deltaOmega > 0 : 
            S_ = self.PHat[s, a] < 1

            if M0 == 0 :
                S_[unvisitedSucc] = False

            donor = np.argmin(V[Pt[s, a] > 0])
            recipient = np.argmax(V[Pt[s, a] < 1 and S_])

            zeta = min(1 - Pt[s, a, donor], Pt[s, a, recipient], deltaOmega)

            if not findUpper :
                donor, recipient = recipient, donor

            Pt[s, a, donor] -= zeta
            Pt[s, a, recipient] += zeta 

            deltaOmega -= zeta

            if unvisitedSucc[sUpper] :
                M0 -= zeta

        return Pt[s, a]

    def confidenceRadius (self, s, a, delta) :
        """
        Referred to as omega in the DDV paper.
        Some magic function probably used to
        make the PAC guarantees go through.

        Parameters
        ----------
        s : int
            State.
        a : int
            Action.
        delta : float
            A confidence interval parameter.
        """
        top = np.log(2 ** self.mdp.S - 2) - np.log(delta)
        return np.sqrt(2 * top / self.Ntotal[s, a])

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
        S = self.mdp.S
        A = self.mdp.A

        Pu = np.zeros(self.mdp.T.shape)
        Pl = np.zeros(self.mdp.T.shape)

        for s, a in product(range(S), range(A)) :
            Pu[s, a] = self.shiftProbabilityMass(s, a, delta, True)
            Pl[s, a] = self.shiftProbabilityMass(s, a, delta, False)

        self.QUpper = QSolver(self.mdp, Pu, self.QUpper, self.stop)
        self.QLower = QSolver(self.mdp, Pl, self.QLower, self.stop) 

