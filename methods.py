import numpy as np
from sklearn.decomposition import PCA
from kernels import *
import pywt
from scipy.stats import qmc
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

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
        k = rng.integers(0, n_dim)  # 🔧 correction ici
        new_sample[i, k], new_sample[j, k] = new_sample[j, k], new_sample[i, k]
        
        new_score = min_dist(new_sample)
        delta = new_score - best_score
        
        # acceptation (recuit simulé)
        if delta > 0 or np.exp(delta / T) > rng.random():
            best_sample = new_sample
            best_score = new_score
        
        T *= alpha  # refroidissement
    
    return qmc.scale(best_sample, bounds[0], bounds[1])


    # Preprocessing: scale outputs (optional but you used it)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)  # shape (n_train, D)

    # PCA on scaled outputs
    pca = PCA(n_components=n_pc)
    Y_train_pca = pca.fit_transform(y_train_scaled)   # (n_train, n_pc)

    # Save train reconstruction check
    Y_rec_train = scaler_y.inverse_transform(pca.inverse_transform(Y_train_pca))
    print("relative error on TRAIN reconstr:", np.linalg.norm(Y_rec_train - y_train) / np.linalg.norm(y_train))
    print("explained variance sum:", np.sum(pca.explained_variance_ratio_))

    # --- GP predictions for each PC (we build mu (m,n_pc) and covariance matrices C_j (m,m)) ---
    # condMean and condVar must accept Y as (n_train, n_pc) and return:
    # - condMean -> (m, n_pc)
    # - condVar  -> either a list of p covariance matrices (m,m) or a 3D array (n_pc, m, m)
    mu_all = condMean(x_test, x_train, Y_train_pca, kernel, param, RdKernel, "sum")  # should return (m, n_pc)
    # For variance, we'll call condVar for each component to make life simpler and explicit:
    m = x_test.shape[0]
    C_list = []
    for j in range(n_pc):
        # ask condVar with Y = Y_train_pca[:, j] (scalar output)
        Cj = condVar(x_test, x_train, Y_train_pca[:, j], kernel, param, RdKernel, "sum")
        # ensure symmetry + jitter
        Cj = 0.5 * (Cj + Cj.T) + 1e-8 * np.eye(m)
        C_list.append(Cj)

    # mu_all is (m, n_pc)
    # Simulate one sample (or several) of PC trajectories consistently:
    PC_sample = np.zeros_like(mu_all)  # (m, n_pc)
    for j in range(n_pc):
        mu_j = mu_all[:, j]
        Cj = C_list[j]
        PC_sample[:, j] = np.random.multivariate_normal(mu_j, Cj)

    # Reconstruct in scaled output space then inverse scale
    Y_sample_scaled = pca.inverse_transform(PC_sample)      # (m, D) in scaled space
    Y_sample = scaler_y.inverse_transform(Y_sample_scaled)  # back to original units

    return Y_sample, mu_all, C_list, pca, scaler_y

def ACP(x_train,x_test,y_train,n_pc,param,kernel):
    #Scaling des données
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    #ACP
    pca = PCA(n_components=n_pc)
    Y_train_pca = pca.fit_transform(y_train_scaled)

    print("Variance expliquée par les 5 premières composantes :",pca.explained_variance_ratio_)
    print("Variance globale expliquée :",np.sum(pca.explained_variance_ratio_))
    print("Taille du jeu d'entrainement transformé par ACP :", Y_train_pca.shape)

    #Un SEUL métamodèle 
    Y_mean = condMean(x_test,x_train,Y_train_pca,kernel,param,RdKernel,"sum")
    Y_var = condVar(x_test,x_train,Y_train_pca,kernel,param,RdKernel,"sum")
    #Simulation 
    PC_test_reconstruct = np.zeros_like(Y_mean)
    for i in range (n_pc):
      PC_test_reconstruct[:,i] = np.random.multivariate_normal(Y_mean[:,i],Y_var)
    #Reconstruction
    #Y_test_reconstruct = scaler_y.inverse_transform(pca.inverse_transform(Y_mean))
    Y_test_reconstruct = scaler_y.inverse_transform(pca.inverse_transform(PC_test_reconstruct))
    #V_sq = V**2 
    #Var_Y_reconstruct = Y_var @ V_sq.T 
    
    return Y_test_reconstruct,None

def ACPF_Ondelettes(x_train,x_test,y_train,n_pc,param,kernel,K_tilde=0,p=0):
    
    #Décomposition en ondelettes 
    n_samples, signal_length = y_train.shape
    # Première décomposition pour avoir les dimensions de sorties
    cA0, cD0 = pywt.dwt(y_train[0, :], wavelet="db4", mode="periodization") #cA = coeff approximation, cD = coeff details
    n_cA = cA0.shape[0]
    n_cD = cD0.shape[0]
    K = n_cA + n_cD

    coeffs_wavelets = np.zeros((n_samples, K), dtype=float)

    for i in range(n_samples):
        cA, cD = pywt.dwt(y_train[i, :], wavelet="db4", mode="periodization")
        coeffs_wavelets[i, :n_cA] = cA
        coeffs_wavelets[i, n_cA:] = cD
    
    #Sélection des K_tildes coefficients pour l'ACP
    lambda_k = np.mean(coeffs_wavelets**2/np.sum(coeffs_wavelets**2,axis=1,keepdims=True),axis=0)
    indices_sorted = np.argsort(lambda_k)[::-1]
    lambda_k_sorted = lambda_k[indices_sorted]
    if K_tilde!=0 :
        indices_ACP = indices_sorted[:K_tilde]
        energy = np.sum(lambda_k_sorted[:K_tilde])
        print(f"Proportion moyenne de l'énergie : {energy} ")
    elif p!=0 : 
        cum_energy = np.cumsum(lambda_k_sorted)
        K_tilde = np.where(cum_energy > p )[0][0]-1
        indices_ACP = indices_sorted[:K_tilde]
        print(f"Nombre de coefficients conservés pour l'ACP : {K_tilde-1}")
    else :
        RuntimeError("Either K_tilde or p must be different of 0")
    coeffs_wavelets_ACP = coeffs_wavelets[:,indices_ACP]
    indices_mean = np.delete(np.arange(K), indices_ACP)
    coeffs_wavelets_mean = coeffs_wavelets[:,indices_mean]

    #ACP sur les coefficients d'ondelettes sélectionnés
    wavelets_test_reconstruct,Var_wavelets_reconstruct = ACP(x_train,x_test,coeffs_wavelets_ACP,n_pc,param,kernel)

    #Moyenne empirique pour les coefficients non sélectionnés
    coeffs_wavelets_mean_reconstruct = np.mean(coeffs_wavelets_mean,axis=0,keepdims=True)

    #Reconstruction de la décomposition en ondelettes pour le jeu de test
    n_test = x_test.shape[0]
    wavelets_test_reconstruct_total = np.zeros((n_test, K), dtype=float)
    wavelets_test_reconstruct_total[:,indices_ACP] = wavelets_test_reconstruct
    wavelets_test_reconstruct_total[:, indices_mean] = coeffs_wavelets_mean_reconstruct

    #Transformée en ondelette inverse pour revenir dans l'espace de départ
    Y_test_reconstruct = np.zeros((n_test, signal_length), dtype=float)

    for i in range(n_test):
        cA = wavelets_test_reconstruct_total[i, :n_cA].copy()
        cD = wavelets_test_reconstruct_total[i, n_cA:n_cA+n_cD].copy()
        y_rec = pywt.idwt(cA, cD, wavelet="db4", mode="periodization")
        # idwt may return a float array slightly different length depending on padding; trim/pad to original length
        if y_rec.shape[0] > signal_length:
            y_rec = y_rec[:signal_length]
            print("A")
        elif y_rec.shape[0] < signal_length:
            y_rec = np.pad(y_rec, (0, signal_length - y_rec.shape[0]), mode='constant')
            print("B")
        Y_test_reconstruct[i, :] = y_rec

    return Y_test_reconstruct

