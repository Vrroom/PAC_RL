from MDP import *
from DDV import *
import numpy as np

mdp = MDPfromJson('./testMDPs/mdp-riverswim.json')
algo = DDV(mdp, 0.1, 0.1)
print(np.argmax(algo.QUpper, axis=1))
