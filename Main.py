from MDP import *
from DDV import *
from BellmanEquation import *
import numpy as np

mdp = MDPfromJson('./testMDPs/mdp-riverswim.json')
stop = lambda i, e : i > 200 
q = QSolver(mdp, mdp.T, np.zeros(mdp.R.shape), stop)
print(q)

print(np.argmax(q, axis=1))
algo = DDV(mdp, 3e-2, 0.1, 1)
print(np.argmax(algo.QUpper, axis=1))
