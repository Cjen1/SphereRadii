from scipy.optimize import linprog
from scipy.spatial import KDTree
from scipy.sparse import dok_matrix as sparse
import numpy as np

# Data input in the form: 
# [node_id, time_step, (x/y/z)]

# Assumption: points marked with Nan are not in existance

def find_pairs_within_d(X, r):
    npoints = X.shape[0]
    ntimes = X.shape[1]

    #This should hopefully be ~(npoints * 20)
    pairs = [{} for _ in range(npoints)]
    for time in range(ntimes):
        tree = KDTree(X[:,time,:])
        for i, js in enumerate(tree.query_ball_tree(tree, r)):
            pairs[i].update(js)

    return pairs

# I'd expect this one to be much much slower
def find_k_nearest_pairs(X, k):
    npoints = X.shape[0]
    ntimes = X.shape[1]
    
    pairs = [{} for _ in range(npoints)]
    for time in range(ntimes):
        tree = KDTree(X[:,time,:])
        for i in range(npoints):
            p = X[i,time,:]
            _, js = tree.query(p, k)
            pairs[i].update(js)

    return pairs

def find_pair_dists(X, r):
    npoints = X.shape[0]
    pairs = find_pairs_dists(X,r)

    pair_dists = sparse((npoints, npoints))
    for i, js in enumerate(pairs):
        # This surprisingly works!
        Xjs = X[js,:,:] # [js,t,3]
        Xi = X[i,:,:] # [1,t,3]

        diff = Xjs - Xi #[js,t,3]
        e_dist = sum(np.power(diff,2), 2) #[js,t,1]

        # This can be swapped for something fancier which takes variation in measuring into account
        min_dist = np.min(e_dist, axis=1)

        pair_dists[i,js] = min_dist

    return pair_dists

def get_radii(X,r):
    npoints = X.shape[0]
    pair_dists = find_pair_dists(X,r)

    f = np.zeros((npoints, 1))

    ii = [i for (i,j), d in pair_dists.items()]
    jj = [j for (i,j), d in pair_dists.items()]
    dd = [d for (i,j), d in pair_dists.items()]

    f[list(dist.fromkeys(ii + jj))] = -1

    n_constraints = len(ii)
    Aub = sparse((n_constraints, n_points))

    cons = [[i,j] for (i,j), _ in pair_dists.items()]
    Aub[range(n_points), cons] = 1

    bub = dd

    res = linprog(f, A_ub = Aub, b_ub = bub)

    return res.x
