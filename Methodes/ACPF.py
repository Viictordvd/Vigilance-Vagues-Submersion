import pywt
import numpy as np
from sklearn.decomposition import PCA

import Methodes.Gaussian_Processes as gp

def ACP_train(x_train,y_train,n_pc,param,kernel_fn=None,verbose=False):
    #Centrage des données
    y_bar = np.mean(y_train,axis=0, keepdims=True)
    y_train_norm = y_train - y_bar
    
    #ACP
    pca = PCA(n_components=n_pc)
    Y_train_pca = pca.fit_transform(y_train_norm)
    V = pca.components_.T #Matrice de projection de l'ACP
    
    models = gp.GP_train(x_train,Y_train_pca,n_pc,param,kernel_fn=None,verbose=verbose) #Entrainement GP sur les composantes principales
    
    print("--- Analyse en Composantes Principales ---")
    print("Variance expliquée par les 5 premières composantes :",pca.explained_variance_ratio_)
    print("Variance globale expliquée :",np.sum(pca.explained_variance_ratio_))
    print("Taille du jeu d'entrainement transformé par ACP :", Y_train_pca.shape)
    
    return models,V,y_bar

def ACP_predict(models,x_test,n_pc,V,y_bar):
    Y_mean_GP = gp.GP_predict(models,x_test,n_pc) #Prédiction par GP
    Y_test_reconstruct = Y_mean_GP @ V.T + y_bar  #Reconstruction
    return Y_test_reconstruct

def bspline_basis_matrices(noeuds, domaine, degree=1):
    def B(i, xx, tt): #Évalue la B-spline linéaire B_i(xx) pour un vecteur de noeuds tt
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
    nBx = len(noeuds[0]) - degree - 1
    Bd = np.array([B(j, domaine[0], noeuds[0]) for j in range(nBx)]).T
    for i in range(1,len(noeuds)):
        t=noeuds[i]
        x=domaine[i]
        nBt = len(t) - degree - 1  
        Bt = np.array([B(i, x, t) for i in range(nBt)]).T
        Bd = np.kron(Bd, Bt)
    return Bd

def B_Splines_train(x_train, y_train,noeuds,domaine,n_pc, param, degree=1,kernel_fn=None,verbose=False):
    B = bspline_basis_matrices(noeuds,domaine,degree=degree)
    
    # calcul des coefficients C par moindres carrés
    C = np.linalg.lstsq(B, y_train.T, rcond=None)[0].T # solve B @ C = y  -> c = lstsq(B, y)
    
    models, V, y_bar = ACP_train(x_train,C,n_pc,param,kernel_fn=None,verbose=False) #ACP sur les coefficients C puis reconstruction (en utilisant les processus gaussiens)
    return models, V, y_bar, B

def B_Splines_predict(models,x_test, n_pc,V,y_bar,Bxy):
    C_reconstruct = ACP_predict(models,x_test,n_pc,V,y_bar) # reconstruction dans l'espace de départ
    Y_test_reconstruct = C_reconstruct @ Bxy.T              # (n_test, n_points)
    return Y_test_reconstruct

def Ondelettes_train(x_train,y_train,n_pc,param,K_tilde=0,p=0,J=1,kernel_fn=None,verbose=False):
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
    models,V,y_bar = ACP_train(x_train,coeffs_wavelets_ACP,n_pc,param,kernel_fn=None,verbose=False)
    return models,V, y_bar, coeffs_wavelets_mean ,coeffs_shapes, signal_length ,indices_ACP, indices_mean

def Ondelettes_predict(models,x_test,n_pc,V, y_bar, coeffs_wavelets_mean ,coeffs_shapes, signal_length ,indices_ACP, indices_mean):
    wavelet = "db4"
    #ACP sur les coefficients d'ondelettes sélectionnés
    wavelets_test_reconstruct = ACP_predict(models, x_test,n_pc,V,y_bar)

    #Moyenne empirique pour les coefficients non sélectionnés
    coeffs_wavelets_mean_reconstruct = np.mean(coeffs_wavelets_mean,axis=0,keepdims=True)

    #Reconstruction de la décomposition en ondelettes pour le jeu de test
    n_test = x_test.shape[0]
    K = sum(coeffs_shapes)
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