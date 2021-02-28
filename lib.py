from scipy.optimize import linprog
from scipy.spatial import KDTree
from scipy.sparse import dok_matrix as sparse
import numpy as np

# Data input in the form: 
# [node_id, time_step, (x/y/z)]

# Assumption: points marked with Nan are not in existance

# We one of thse two 
def find_pairs_within_d(X, r):
    npoints = X.shape[0]
    ntimes = X.shape[1]

    pairs = [set() for _ in range(npoints)]
    for time in range(ntimes):
        tree = KDTree(X[:,time,:])
        for i, js in enumerate(tree.query_ball_tree(tree, r)):
            js = [j for j in js if j != i]
            pairs[i] = pairs[i].union(js)

    return pairs

# I'd expect this one to be much much slower
def find_k_nearest_pairs(X, k):
    npoints = X.shape[0]
    ntimes = X.shape[1]
    
    pairs = [set() for _ in range(npoints)]
    for time in range(ntimes):
        tree = KDTree(X[:,time,:])
        for i in range(npoints):
            p = X[i,time,:]
            _, js = tree.query(p, k + 1)
            js = [j for j in js if j != i]
            pairs[i] = pairs[i].union(js)

    return pairs

# Return the pair distances of particles within r of each other at any point in time
# Only returns the lexographically least pair distance
def find_pair_dists(X, r):
    npoints = X.shape[0]
    pairs = find_pairs_within_d(X,r)

    pair_dists = sparse((npoints, npoints))
    for i, js in enumerate(pairs):
        js = list(js)
        # Ensures that we don't emit too many constraints
        js = [j for j in js if i < j]

        # Indexing by js surprisingly works!
        Xjs = X[js,:,:] # [js,t,3]
        Xi = X[i,:,:] # [1,t,3]

        diff = Xjs - Xi #[js,t,3]
        e_dist = np.sqrt(np.sum(np.power(diff,2), 2)) #[js,t,1]

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

    f[list(dict.fromkeys(ii + jj))] = -1

    n_constraints = len(ii)
    Aub = sparse((n_constraints, npoints))

    links = [(i,j) for (i,j), _ in pair_dists.items()]
    for line, (i,j) in enumerate(links):
        Aub[line, [i,j]] = 1

    bub = dd

    res = linprog(f, A_ub = Aub, b_ub = bub)

    return res.x
