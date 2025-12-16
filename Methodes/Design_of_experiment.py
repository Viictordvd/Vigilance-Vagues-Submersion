import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist

#Hyperoptimized Latin Hypercube Sampling
def lhs_optimized(n_samples, n_dim,bounds, n_iter=1000, seed=None):
    #bounds : array of dimension (2 x n_dim) with lower and upper bounds
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
    sample = sampler.random(n=n_samples)
    
    def min_dist(X):
        # critère maximin : plus grande distance minimale possible
        return np.min(pdist(X))
    
    best_sample = sample.copy()
    best_score = min_dist(best_sample)
    
    T0 = 1.0  # température initiale
    alpha = 0.99  # taux de refroidissement
    T = T0
    
    for _ in range(n_iter):
        # échange de deux valeurs dans une dimension aléatoire
        new_sample = best_sample.copy()
        i, j = rng.integers(0, n_samples, 2)
        k = rng.integers(0, n_dim)  
        new_sample[i, k], new_sample[j, k] = new_sample[j, k], new_sample[i, k]
        
        new_score = min_dist(new_sample)
        delta = new_score - best_score
        
        # acceptation (recuit simulé)
        if delta > 0 or np.exp(delta / T) > rng.random():
            best_sample = new_sample
            best_score = new_score
        
        T *= alpha  # refroidissement
    
    return qmc.scale(best_sample, bounds[0], bounds[1])