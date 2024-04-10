import copy
import numpy as np
from pathlib import Path
import sys
root = Path(__file__).parent.parent
sys.path.append(str(root))
from env.grid_scenarios import MiniWorld
P = [[0 for j in range(4)] for i in range(36)]
print(type(P))
for i in range(4):
    P[0][i]= [(1, 0, -0.1, False)]
print(P)
done = False
print(1-False)
print("*"*20)
l = [1,2,3,4]
r = l.reverse()
#r = reversed(l)
print(r)