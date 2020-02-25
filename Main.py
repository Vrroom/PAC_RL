from MDP import *
from DDV import *
from MBIE import *
from BellmanEquation import *
import numpy as np

# mdp = randomMDP(4, 2)
mdp = MDPfromJson('./testMDPs/mdp-riverf.json')
stop = lambda i, e : i > 200 
q = QSolver(mdp, mdp.T, np.zeros(mdp.R.shape), stop)
print(np.argmax(q, axis=1))
algo = MBIE(mdp, 0.1, 0.1, 1)
print(np.argmax(algo.QUpper, axis=1))
