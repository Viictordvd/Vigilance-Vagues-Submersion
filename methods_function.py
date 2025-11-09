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

def L2_norm(f,g,t):
    dt=t[1]-t[0]
    return np.sum((f(t)-g(t))**2)*dt

class FunctionalL2Kernel(gpflow.kernels.Kernel):
    """
    Noyau k(X,Y) = σ² exp(-Σ_i ||f_i - g_i||_L2²)
    où chaque X[i] est un ensemble de fonctions fi(t) discrétisées.
    """
    def __init__(self, variance=1.0,lengthscale=1.0):
        super().__init__()
        self.lengthscale = gpflow.Parameter(lengthscale, transform=gpflow.utilities.positive())
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive())

    def _L2_distance(self, F, G):
        # F et G ont shape (n_points_t, n_funcs=8)
        diff = F - G
        dt = 1.0 / tf.cast(tf.shape(F)[0], tf.float64)
        # ∑₁⁸ ∫ (f_i - g_i)² ≈ ∑₁⁸ ∑ₜ (f_i(t)-g_i(t))² * dt
        return tf.reduce_sum(diff**2) * dt

    def K(self, X, Y=None):
        if Y is None:
            Y = X

        X = tf.reshape(X, (tf.shape(X)[0], 8, -1))
        Y = tf.reshape(Y, (tf.shape(Y)[0], 8, -1))

        X_exp = tf.expand_dims(X, 1)  # [N,1,8,n_t]
        Y_exp = tf.expand_dims(Y, 0)  # [1,M,8,n_t]
        diff = X_exp - Y_exp
        dt = 1.0 / tf.cast(tf.shape(X)[-1], tf.float64)
        l2 = tf.reduce_sum(diff**2, axis=[2,3]) * dt  # [N,M]
        return (self.variance**2) * tf.exp(-l2 / (self.lengthscale**2))

    def K_diag(self, X):
        return tf.fill([tf.shape(X)[0]], tf.squeeze(self.variance ** 2))


def GP(x_train, x_test, y_train, t, n_pc, param):
    means = []
    
    X_train = np.zeros((x_train.shape[0],x_train.shape[1],len(t)))
    X_test = np.zeros((x_test.shape[0],x_test.shape[1],len(t)))
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            X_train [i,j,:] = x_train[i,j](t)
    for i in range(x_test.shape[0]):
        for j in range(x_test.shape[1]):
            X_test [i,j,:] = x_test[i,j](t)
                  
    # Conversion numpy -> tensorflow
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    X_test_tf  = tf.convert_to_tensor(X_test, dtype=tf.float64)
    
    X_train_flat = tf.reshape(X_train_tf, (X_train_tf.shape[0], -1))
    X_test_flat  = tf.reshape(X_test_tf,  (X_test_tf.shape[0],  -1))

    for i in range(n_pc):
        print("\n--- Entraînement du modèle GP pour la composante principale ", i+1, "---")
        Y_train = tf.convert_to_tensor(y_train[:, i:i+1], dtype=tf.float64)
        
        # Définition du kernel : (variance * RBF(length_scale))
        kernel = FunctionalL2Kernel(lengthscale=param[0], variance=param[1]**2)
        
        # Modèle GP régressif
        model = gpflow.models.GPR(data=(X_train_flat, Y_train),kernel=kernel,mean_function=Constant())
        print("Modèle GP créé pour la composante principale ", i+1)
        # Optimisation des hyperparamètres
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
        
        # Affichage des hyperparamètres optimisés
        print("Affichage des hyperparamètres optimisés")
        print(f"\n--- Composante principale {i+1} ---")
        gpflow.utilities.print_summary(model)
        
        # Prédiction
        mean_i, var_i = model.predict_f(X_test_flat)
        means.append(mean_i.numpy().flatten())
        print("Prédiction effectuée pour la composante principale ", i+1)

    mean = np.column_stack(means)
    return mean

def ACP(x_train,x_test,y_train,t,n_pc,param):
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
    Y_mean_GP = GP(x_train,x_test,Y_train_pca,t,n_pc,param)
    
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

def Bsplines_ACP(x_train, x_test, y_train,t1, t2, t, n_pc, param, degree=1):
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
    C_reconstruct = ACP(x_train,x_test,C,t,n_pc,param)
    print("taille du vecteur de C_reconstruct:",C_reconstruct.shape)
    
    # reconstruction dans l'espace de départ
    Y_test_reconstruct = C_reconstruct @ Bxy.T   # (n_test, n_points)
    print("taille du vecteur de Y_test_reconstruct:",Y_test_reconstruct.shape)

    return Y_test_reconstruct

def ACPF_Ondelettes(x_train,x_test,y_train,t,n_pc,param,K_tilde=0,p=0,J=1):

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
    wavelets_test_reconstruct = ACP(x_train,x_test,coeffs_wavelets_ACP,t,n_pc,param)

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