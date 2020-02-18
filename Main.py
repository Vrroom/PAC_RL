from MDP import *
from DDV import *
import numpy as np

mdp = MDPfromJson('./testMDPs/mdp-riverswim.json')
algo = DDV(mdp, 1e-3, 0.1, 123124)
print(np.argmax(algo.QUpper, axis=1))
