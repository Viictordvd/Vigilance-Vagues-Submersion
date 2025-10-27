import numpy as np
from sklearn.decomposition import PCA
from kernels import *
import pywt

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

def ACPF_Ondelettes(x_train,x_test,y_train,n_pc,param,kernel,K_tilde=0,p=0):
    
    #Décomposition en ondelettes 
    n_samples, signal_length = y_train.shape
    coeffs_list = []
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
        coeffs_list.append((cA, cD))

        shapes = {'cA_length': n_cA, 'cD_length': n_cD, 'signal_length': signal_length}
    
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

    #ACP sur les coefficients d'ondelettes
    pca = PCA(n_components=n_pc)
    wavelets_bar = np.mean(coeffs_wavelets_ACP,axis=0)
    wavelets_train_norm = coeffs_wavelets_ACP - wavelets_bar
    wavelets_train_pca = pca.fit_transform(wavelets_train_norm)
    V = pca.components_.T #Matrice de projection de l'ACP

    #Régression par GP sur les composantes principales
    print("Variance expliquée par les 5 premières composantes :",pca.explained_variance_ratio_)
    print("Variance globale expliquée :",np.sum(pca.explained_variance_ratio_))
    print("Taille du jeu d'entrainement transformé par ACP :", wavelets_train_pca.shape)
    wavelets_mean = np.zeros((n_pc,x_test.shape[0]))
    wavelets_var = np.zeros((n_pc,x_test.shape[0],x_test.shape[0]))

    #Un métamodèle par composante principale
    for i in range(n_pc):
        wavelets_mean[i,:] = condMean(x_test,x_train,wavelets_train_pca[:,i],RdKernel,param,kernel,"sum")
        wavelets_var [i,:,:] = condVar(x_test,x_train,wavelets_train_pca[:,i],RdKernel,param,kernel,"sum")

    #Reconstruction
    wavelets_test_reconstruct = wavelets_mean.T @ V.T + wavelets_bar[None,:]
    Var_wavelets_PC_reconstruct = np.stack([np.diag(Cj) for Cj in wavelets_var], axis=1)
    V_sq = V**2 
    Var_wavelets_reconstruct = Var_wavelets_PC_reconstruct @ V_sq.T 

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
        elif y_rec.shape[0] < signal_length:
            y_rec = np.pad(y_rec, (0, signal_length - y_rec.shape[0]), mode='constant')
        Y_test_reconstruct[i, :] = y_rec

    return Y_test_reconstruct

