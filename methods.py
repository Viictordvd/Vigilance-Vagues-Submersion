import numpy as np
from sklearn.decomposition import PCA
from kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def GP(x_train,x_test,y_train,n_pc,param):
    #Prédiction GP sur les composantes principales
    means = []
    variances = []
    for i in range(n_pc):
        kernel_gp = (param[1]**2) * RBF(length_scale=param[0]) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel_gp, normalize_y=True)
        gp.fit(x_train, y_train[:, i])
        mean_i, std_i = gp.predict(x_test, return_std=True)
        means.append(mean_i)
        variances.append(std_i**2)  # variance = std²

    mean = np.column_stack(means)
    var = np.column_stack(variances)
    return mean#,var
 
def ACP(x_train,x_test,y_train,n_pc,param):
    print("taille du vecteur de y_train:",y_train.shape)
    #ACP
    pca = PCA(n_components=n_pc)
    y_bar = np.mean(y_train,axis=0, keepdims=True)
    y_train_norm = y_train - y_bar
    Y_train_pca = pca.fit_transform(y_train_norm)
    
    #Matrice de projection de l'ACP
    V = pca.components_.T
    
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
    C_bar = np.mean(C,axis=0, keepdims=True)
    C_centered = C-C_bar
    print("taille du vecteur de coefficients C centré:",C_centered.shape)
    
    #PCA sur les coefficients
    pca = PCA(n_components=n_pc)
    C_train_pca = pca.fit_transform(C_centered)
    print("taille de la c_train pca :",C_train_pca.shape)
    #Matrice de projection de l'ACP
    V = pca.components_.T                  
    
    print("---")
    print("Variance expliquée par les 5 premières composantes :",pca.explained_variance_ratio_)
    print("Variance globale expliquée :",np.sum(pca.explained_variance_ratio_))
    print("---")
    print("Taille du jeu d'entrainement transformé par ACP :", C_train_pca.shape)
    print("taille du vecteur de coefficients après l'ACP:",V.shape)

    #Prédiction GP sur les composantes principales
    C_mean_GP = GP(x_train,x_test,C_train_pca,n_pc,param)
    print("taille de C_mean_GP:",C_mean_GP.shape)
    
    # Reconstruction : moyenne des coefficients pour chaque test
    C_reconstruct = C_mean_GP @ V.T + C_bar    # (n_test, n_basis)
    print("taille du vecteur de C_reconstruct:",C_reconstruct.shape)
    
    # reconstruction dans l'espace de départ
    Y_test_reconstruct = C_reconstruct @ Bxy.T   # (n_test, n_points)
    print("taille du vecteur de Y_test_reconstruct:",Y_test_reconstruct.shape)
    
    return Y_test_reconstruct