import numpy as np
import random

X = [1]
i = 1
flag = True
p = 0.4
N = 10
a = 1
b = 100
Q = 100
X.append(np.random.uniform(a,b))

def generate_speed_cost(speed): return 0.05 * speed + 1

def z12(X, dd, ll, Q):
    indp = 1
    X.pop(0)
    velp = X[0]
    X.pop(0)
    ind = 2
    sm = 0
    vh_load = ll[0]
    z2 = 0
    cc = True
    if vh_load < 0 or vh_load > Q: cc = False
    for ii, elem in enumerate(X):
        if ii%2:
            dist = 0
            for k in range(indp, ind):
                dist += dd[k-1]
            sm += dist/velp

            z2 += dist * vh_load * generate_speed_cost(velp)

            velp = elem
            vh_load += ll[ind-1]

            if vh_load < 0 or vh_load > Q: cc = False
        else:
            ind = elem

    return sm, z2, cc

N_total_nodes = 100

for _ in range(N):

    random_value = random.randrange(k)
    i += random_value + 1
    X.append(i)
    X.append(np.random.uniform(a,b))

dd = []
dd_total = 0

ll = []
dd_min = 10
dd_max = 1000

ll_min = -50
ll_max = 50
for _ in range(N_total_nodes):
    daux = np.random.uniform(dd_min, dd_max)
    dd.append(daux)
    ll.append(np.random.uniform(ll_min, ll_max))
    dd_total += daux


print(X)
