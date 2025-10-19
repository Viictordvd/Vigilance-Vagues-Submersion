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