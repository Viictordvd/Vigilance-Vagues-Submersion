import numpy as np
from sklearn.decomposition import PCA
from kernels import *
import tensorflow as tf
import gpflow
from gpflow.kernels import SquaredExponential
from gpflow.mean_functions import Constant
import pywt
from scipy.stats import qmc
from scipy.spatial.distance import pdist

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


def GP(x_train, x_test, y_train, n_pc, param):
    means = []

    # Conversion numpy -> tensorflow
    X_train = tf.convert_to_tensor(x_train, dtype=tf.float64)
    X_test  = tf.convert_to_tensor(x_test, dtype=tf.float64)

    for i in range(n_pc):
        Y_train = tf.convert_to_tensor(y_train[:, i:i+1], dtype=tf.float64)

        # Définition du kernel : (variance * RBF(length_scale)) + bruit blanc
        kernel = gpflow.kernels.SquaredExponential(lengthscales=param[0], variance=param[1]**2) + gpflow.kernels.White(variance=1e-6)

        # Modèle GP régressif
        model = gpflow.models.GPR(data=(X_train, Y_train),kernel=kernel,mean_function=Constant())

        # Optimisation des hyperparamètres
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
        
        # Affichage des hyperparamètres optimisés
        print("Affichage des hyperparamètres optimisés")
        print(f"\n--- Composante principale {i+1} ---")
        gpflow.utilities.print_summary(model)
        
        # Prédiction
        mean_i, var_i = model.predict_f(X_test)
        means.append(mean_i.numpy().flatten())

    mean = np.column_stack(means)
    return mean

def ACP(x_train,x_test,y_train,n_pc,param):
    #Centrage des données
    y_bar = np.mean(y_train,axis=0, keepdims=True)
    y_train_norm = y_train - y_bar
    #ACP
    pca = PCA(n_components=n_pc)
    Y_train_pca = pca.fit_transform(y_train_norm)
    
    #Matrice de projection de l'ACP
    V = pca.components_.T
    print(Y_train_pca.shape)
    print("Variance expliquée par les 5 premières composantes :",pca.explained_variance_ratio_)
    print("Variance globale expliquée :",np.sum(pca.explained_variance_ratio_))
    print("Taille du jeu d'entrainement transformé par ACP :", Y_train_pca.shape)
    
    #Prédiction GP sur les composantes principales
    Y_mean_GP = GP(x_train,x_test,Y_train_pca,n_pc,param)
    
    #Reconstruction
    Y_test_reconstruct = Y_mean_GP @ V.T + y_bar
    print("taille du vecteur de Y_test_reconstruct:",Y_test_reconstruct.shape)
    
    return Y_test_reconstruct

def bspline_basis_matrices(t1, t2, x, y, degree=1):
    #Évalue la B-spline linéaire B_i(xx) pour un vecteur de noeuds tt
    def B(i, xx, tt):
        yy = np.zeros_like(xx)
        if tt[i+1] > tt[i]:
            mask1 = (xx >= tt[i]) & (xx < tt[i+1])
            yy[mask1] = (xx[mask1] - tt[i]) / (tt[i+1] - tt[i])
        if tt[i+2] > tt[i+1]:
            mask2 = (xx >= tt[i+1]) & (xx < tt[i+2])
            yy[mask2] = (tt[i+2] - xx[mask2]) / (tt[i+2] - tt[i+1])
        if i == len(tt) - 3 and np.any(xx == tt[-1]):
            yy[xx == tt[-1]] = 1
        return yy

    #Matrices 1D
    nBx = len(t1) - degree - 1
    nBy = len(t2) - degree - 1
    Bx = np.zeros((len(x), nBx))
    By = np.zeros((len(y), nBy))

    for i in range(nBx):
        Bx[:, i] = B(i, x, t1)
    for j in range(nBy):
        By[:, j] = B(j, y, t2)

    #Matrice tensorielle 2D (produit de Kronecker)
    Bxy = np.kron(By, Bx)

    return Bx, By, Bxy

def Bsplines_ACP(x_train, x_test, y_train,t1, t2, n_pc, param, degree=1):
    print("taille du vecteur de y_train:",y_train.shape)
    n_points = y_train.shape[1]   # ex: 4096
    
    #construire la grille 1D (on suppose grille régulière sur [-90,90] pour Z1,Z2)
    n_grid = int(np.sqrt(n_points))
    assert n_grid * n_grid == n_points, "y_train doit correspondre à une grille carrée"
    z = np.linspace(-90, 90, n_grid)

    #construire matrices B-spline
    _, _, Bxy = bspline_basis_matrices(t1, t2, z, z, degree=degree)
    
    # Bxy : (n_points, n_basis)
    n_basis = Bxy.shape[1]
    print("taille de la base B-spline :",n_basis)

    # calcul des coefficients C par moindres carrés
    # solve Bxy @ c = y  -> c = lstsq(Bxy, y)
    # y_train.T shape (n_points, n_train) -> lstsq returns (n_basis, n_train)
    C = np.linalg.lstsq(Bxy, y_train.T, rcond=None)[0].T
    
    #ACP sur les coefficients C puis reconstruction (en utilisant les processus gaussiens)
    C_reconstruct = ACP(x_train,x_test,C,n_pc,param)
    print("taille du vecteur de C_reconstruct:",C_reconstruct.shape)
    
    # reconstruction dans l'espace de départ
    Y_test_reconstruct = C_reconstruct @ Bxy.T   # (n_test, n_points)
    print("taille du vecteur de Y_test_reconstruct:",Y_test_reconstruct.shape)

    return Y_test_reconstruct

def ACPF_Ondelettes(x_train,x_test,y_train,n_pc,param,K_tilde=0,p=0,J=1):

    n_samples, signal_length = y_train.shape
    wavelet = "db4"

    # Décomposition en ondelettes multi-résolution (J niveaux)
    coeffs_list = []
    for i in range(n_samples):
        coeffs = pywt.wavedec(y_train[i, :], wavelet=wavelet, mode="periodization", level=J)
        coeffs_list.append(coeffs)

    # Structure des coefficients : [cA_J, cD_J, cD_{J-1}, ..., cD_1]
    coeffs_shapes = [c.shape[0] for c in coeffs_list[0]]
    K = sum(coeffs_shapes)
    coeffs_wavelets = np.zeros((n_samples, K))
    for i, coeffs in enumerate(coeffs_list):
        coeffs_wavelets[i, :] = np.concatenate(coeffs)

    #Sélection des K_tildes coefficients pour l'ACP
    #Calcul du ratio d'énergie moyen de chaque coefficient
    lambda_k = np.mean(coeffs_wavelets**2/np.sum(coeffs_wavelets**2,axis=1,keepdims=True),axis=0)
    #Tri dans l'ordre décroissant
    indices_sorted = np.argsort(lambda_k)[::-1]
    lambda_k_sorted = lambda_k[indices_sorted]
    if K_tilde!=0 :
        #K_tilde !=0 , on prend les K_tilde coefficients d'ondelettes avec lambda_k les plus élevés
        indices_ACP = indices_sorted[:K_tilde]
        energy = np.sum(lambda_k_sorted[:K_tilde])
        print(f"Proportion moyenne de l'énergie : {energy} ")
    elif p!=0 : 
        # p!=0, on prend les coefficients d'ondelettes avec lambda_k les plus élevés jusquà ce que la somme des lambda_k soit supérieure à p
        K_tilde = np.searchsorted(np.cumsum(lambda_k_sorted), p, side='left') + 1
        indices_ACP = indices_sorted[:K_tilde]
        print(f"Nombre de coefficients conservés pour l'ACP : {K_tilde-1}")
    else :
        ValueError("Either K_tilde or p must be different of 0")
    #Séparation des deux types de coefficients d'ondelettes : ceux prédit par ACP et ceux prédit par moyenne empirique
    indices_ACP.sort()
    indices_mean = np.setdiff1d(np.arange(K), indices_ACP)
    coeffs_wavelets_ACP = coeffs_wavelets[:,indices_ACP]
    coeffs_wavelets_mean = coeffs_wavelets[:,indices_mean]

    #ACP sur les coefficients d'ondelettes sélectionnés
    wavelets_test_reconstruct = ACP(x_train,x_test,coeffs_wavelets_ACP,n_pc,param)

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
        coeffs_flat = wavelets_test_reconstruct_total[i, :]
        coeffs_rec = []
        idx = 0
        # On resépare les coefficients de chaque résolutions
        for shape in coeffs_shapes:
            coeffs_rec.append(coeffs_flat[idx:idx+shape])
            idx += shape
        # Reconstruction inverse multirésolution
        Y_test_reconstruct[i, :] = pywt.waverec(coeffs_rec, wavelet=wavelet, mode="periodization")

    return Y_test_reconstruct