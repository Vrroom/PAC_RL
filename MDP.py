import numpy as np 
import json

def MDPfromJson (filename) :
    """
    Create an MDP object from 
    json file.

    Parameters
    ----------
    filename : str
        Path to file.
    """
    with open (filename, 'r') as fd :
        dct = json.loads(fd.read())

    s0 = int(dct['s0'])
    S = int(dct['S'])
    A = int(dct['A'])
    T = np.array(dct['T'])
    gamma = float(dct['gamma'])
    R = np.array(dct['R'])
    return MDP(s0, S, A, T, gamma, R, 0)

class MDP () :

    def __init__ (self, s0, S, A, T, gamma, R, seed) :
        """
        Constructor
        
        Parameters
        ----------
        s0 : int
            The initial state.
        S : int
            Number of states.
        A : int
            Number of actions.
        T : np.ndarray
            An S by A by S tensor where the
            i, j, k entry tells us the probability
            of transitioning to state k by taking
            action j from state i.
        gamma : float
            Discount factor.
        R : np.ndarray
            An S by A matrix denoting the
            reward associated with picking
            a particular action from a state.
        seed : int
            Seed for the random number
            generator. To ensure 
            reproducibility.
        """
        self.s0 = s0
        self.S = S
        self.A = A
        self.T = T
        self.gamma = gamma
        self.R = R

        self.Rmax = np.max(R)
        self.Rmin = np.min(R)

        # Normalize rewards
        self.R = (self.R - self.Rmin) / (self.Rmax - self.Rmin)
        
        # Calculate the maximum value
        # that can be achieved
        self.Vmax = 1 / (1 - gamma)

        # Random number generator
        self.rng = np.random.RandomState(seed)

    def step(self, s, a) :
        """
        Take a step in the MDP. 
        Given a (state, action), get the 
        next state and reward.

        Parameters
        ----------
        s : int
            State in MDP.
        a : int
            Action.
        """
        s_ = self.rng.choice(self.S, p=self.T[s][a])
        return s_, self.R[s][a]
