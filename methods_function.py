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

def GP(x_train, x_test, y_train, n_pc, param):
    means = []

    # Conversion numpy -> tensorflow
    X_train = tf.convert_to_tensor(x_train, dtype=tf.float64)
    X_test  = tf.convert_to_tensor(x_test, dtype=tf.float64)

    for i in range(n_pc):
        Y_train = tf.convert_to_tensor(y_train[:, i:i+1], dtype=tf.float64)

        # Définition du kernel : (variance * RBF(length_scale)) + bruit blanc
        kernel = gpflow.kernels.SquaredExponential(lengthscales=param[0], variance=param[1]**2)

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