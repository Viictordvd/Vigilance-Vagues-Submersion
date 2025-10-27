import numpy as np
from sklearn.decomposition import PCA
from kernels import *
 
def ACP(x_train,x_test,y_train,n_pc,param,kernel):
    #ACP
    pca = PCA(n_components=n_pc)
    y_bar = np.mean(y_train,axis=0)
    y_train_norm = y_train - y_bar
    Y_train_pca = pca.fit_transform(y_train_norm)
    #Matrice de projection de l'ACP
    V = pca.components_.T
    
    print("Variance expliquée par les 5 premières composantes :",pca.explained_variance_ratio_)
    print("Variance globale expliquée :",np.sum(pca.explained_variance_ratio_))
    print("Taille du jeu d'entrainement transformé par ACP :", Y_train_pca.shape)
    Y_mean = np.zeros((n_pc,x_test.shape[0]))
    Y_var = np.zeros((n_pc,x_test.shape[0],x_test.shape[0]))

    #Un métamodèle par composante principale
    for i in range(n_pc):
        Y_mean[i,:] = condMean(x_test,x_train,Y_train_pca[:,i],RdKernel,param,kernel,"sum")
        Y_var [i,:,:] = condVar(x_test,x_train,Y_train_pca[:,i],RdKernel,param,kernel,"sum")

    #Reconstruction
    Y_test_reconstruct = Y_mean.T @ V.T + y_bar[None,:]
    Var_Y_PC_reconstruct = np.stack([np.diag(Cj) for Cj in Y_var], axis=1)
    V_sq = V**2 
    Var_Y_reconstruct = Var_Y_PC_reconstruct @ V_sq.T 
    
    return Y_test_reconstruct,Var_Y_reconstruct

def bspline_basis_matrices(t1, t2, x, y, degree=1):
    """
    Construit les matrices de base B-spline 1D (Bx, By) et 2D (Bxy)
    à partir des vecteurs de noeuds t1, t2 et des points d'évaluation x, y.
    
    Paramètres
    ----------
    t1, t2 : array_like
        Vecteurs de noeuds (pour x et y).
    x, y : array_like
        Points d'évaluation (dans [0,1]).
    degree : int
        Degré de la B-spline (par défaut = 1).
    
    Retourne
    --------
    Bx : ndarray, shape (len(x), nBx)
        Matrice des fonctions de base 1D en x.
    By : ndarray, shape (len(y), nBy)
        Matrice des fonctions de base 1D en y.
    Bxy : ndarray, shape (len(x)*len(y), nBx*nBy)
        Matrice de base tensorielle 2D (Kronecker product).
    """

    def N(i, xx, tt):
        """Évalue la B-spline linéaire N_i(xx) pour un vecteur de noeuds tt"""
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

    # --- Matrices 1D
    nBx = len(t1) - degree - 1
    nBy = len(t2) - degree - 1
    Bx = np.zeros((len(x), nBx))
    By = np.zeros((len(y), nBy))

    for i in range(nBx):
        Bx[:, i] = N(i, x, t1)
    for j in range(nBy):
        By[:, j] = N(j, y, t2)

    # --- Matrice tensorielle 2D (produit de Kronecker)
    Bxy = np.kron(By, Bx)

    return Bx, By, Bxy

def BsplinesDecomposition(x_train, x_test, y_train, t1, t2, param, kernel):
    """
    Substitue l'ACP par une décomposition B-spline tensorielle pour la métamodélisation de champs 2D.
    """
    #Construction de la base B-spline sur le maillage
    n_grid = int(np.sqrt(y_train.shape[1]))  # taille d’un côté (ex: 64x64)
    z = np.linspace(0, 1, n_grid)
    Bx, By, Bxy = bspline_basis_matrices(t1, t2, z, z)
    n_basis = Bxy.shape[1]
    #Projection des données sur la base (calcul des coefficients)
    Bmat = Bxy  # (n_grid^2, n_basis)
    G = np.linalg.pinv(Bmat)  # pseudo-inverse
    C_train = (G @ y_train.T).T  # (n_train, n_basis)
    
    print(f"Base B-spline : {n_basis} fonctions de base")
    print(f"Coefficients shape : {C_train.shape}")

    #Apprentissage (un modèle par coefficient)
    n_test = x_test.shape[0]
    Y_mean = np.zeros((n_basis, n_test))
    Y_var  = np.zeros((n_basis, n_test, n_test))

    for i in range(n_basis):
        print(f'calcul en cours : {i}/{n_basis}')
        Y_mean[i,:] = condMean(x_test, x_train, C_train[:,i], RdKernel, param, kernel, "sum")
        Y_var[i,:,:] = condVar(x_test, x_train, C_train[:,i], RdKernel, param, kernel, "sum")

    #Reconstruction
    Y_test_reconstruct = (Y_mean.T @ Bmat.T)
    Var_Y_PC_reconstruct = np.stack([np.diag(Cj) for Cj in Y_var], axis=1)
    B_sq = Bmat**2
    Var_Y_reconstruct = Var_Y_PC_reconstruct @ B_sq.T
    
    return Y_test_reconstruct, Var_Y_reconstruct