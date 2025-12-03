import numpy as np
import tensorflow as tf
import gpflow
from gpflow.mean_functions import Constant
from validation_gp import validation_GP

class FunctionalL2Kernel(gpflow.kernels.Kernel):
    """
    Noyau k(X,Y) = σ² exp(-Σ_i ||f_i - g_i||_L2²)
    où chaque X[i] est un ensemble de fonctions fi(t) discrétisées.
    """
    def __init__(self, variance=1.0,lengthscale=1.0, n_entries=8):
        super().__init__()
        self.lengthscale = gpflow.Parameter(lengthscale, transform=gpflow.utilities.positive())
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive())
        self.alpha = gpflow.Parameter(0.1, transform=gpflow.utilities.positive())
        self.n_entries = n_entries
    #Calcul la norme de Sobolev
    def _L2_and_derivative_distance(self, X, Y):
        dX = X[..., 1:] - X[..., :-1]
        dY = Y[..., 1:] - Y[..., :-1]

        dt = 1.0 / tf.cast(tf.shape(X)[-1], tf.float64)

        l2 = tf.reduce_sum((X - Y)**2, axis=[2,3]) * dt
        l2_deriv = tf.reduce_sum((dX - dY)**2, axis=[2,3]) * dt

        return l2 + self.alpha * l2_deriv
    
    def _chunked_K(self, X, Y, chunk_size=50):
        """
        Calcule la matrice de Gram K(X, Y) par morceaux pour éviter
        une surcharge mémoire (OOM) lorsque X est grand.

        On divise X en sous-ensembles ("chunks") de taille chunk_size,
        on calcule la matrice du noyau entre chaque sous-ensemble et Y,
        puis on concatène les résultats.
        
        Entrées :
          - X : tenseur de forme [N, 8 * T] ou [N, 8, T]
          - Y : tenseur de forme [M, 8 * T] ou [M, 8, T]
          - chunk_size : taille des sous-ensembles pour le calcul par morceaux
          
        Sortie :
          - K : matrice de Gram de forme [N, M] avec K_ij = k(X_i, Y_j)
        """
        N = X.shape[0]
        Ks = []
        for i in range(0, N, chunk_size):
            X_chunk = X[i:i+chunk_size]
            Ks.append(self._full_K(X_chunk, Y))
        return tf.concat(Ks, axis=0)

    def _full_K(self, X, Y):
        """
        Calcule la matrice complète K(X, Y) sans découpage.

        Entrées :
          - X : tenseur de forme [N, 8 * T] ou [N, 8, T]
          - Y : tenseur de forme [M, 8 * T] ou [M, 8, T]
        
        Sortie :
          - K : matrice de Gram de forme [N, M] avec K_ij = k(X_i, Y_j)
        """
        # On redimensionne : (N, 8, T)
        X = tf.reshape(X, (tf.shape(X)[0], self.n_entries, -1))
        Y = tf.reshape(Y, (tf.shape(Y)[0], self.n_entries, -1))
        
        # On ajoute des dimensions pour pouvoir diffuser les soustractions :
        #  X_exp : [N, 1, 8, T]
        #  Y_exp : [1, M, 8, T]
        X_exp = tf.expand_dims(X, 1)
        Y_exp = tf.expand_dims(Y, 0)
        
        #On calcule
        dt = 1.0 / tf.cast(tf.shape(X)[-1], tf.float64)                     # Pas de discrétisation temporelle
        l2 = self._L2_and_derivative_distance(X_exp,Y_exp)                  # Calcul de ||X-Y||_L2²
        return (self.variance ** 2) * tf.exp(-l2 / (self.lengthscale ** 2)) # Noyau exponentiel : K = σ² * exp(- ||X-Y||_L2² / ℓ²)
    
    # Fonction principale appelée par gpflow. Renvoie la matrice de Gram K(X, Y). Si Y n'est pas fourni, on calcule K(X, X).
    def K(self, X, Y=None):
        if Y is None:
            Y = X
        return self._chunked_K(X, Y, chunk_size=50)

    # Retourne uniquement la diagonale K(x_i, x_i), qui vaut simplement σ² car ||x_i - x_i||_L2 = 0
    def K_diag(self, X):
        return tf.fill([tf.shape(X)[0]], tf.squeeze(self.variance ** 2))

# Fonction principale pour la régression par processus gaussiens avec entrées fonctionnelles
def GP_train(x_train, y_train, n_pc, param,kernel_fn=None,verbose=False):
    n_entries = x_train.shape[1]
    X_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float64) # Conversion numpy -> tensorflow
    # Aplatissement pour compatibilité avec gpflow, gpflow attend des entrées [N, D], on a donc besoin de "vectoriser" nos fonctions : chaque observation devient un vecteur concaténé contenant toutes les valeurs discrètes des 8 fonctions
    # On reconstruira les fonctions au sein du noyau
    X_train_flat = tf.reshape(X_train_tf, (X_train_tf.shape[0], -1))

    models = [] #Liste des modèles GP de chacune des composantes principales

    for i in range(n_pc):
        print("\n--- Entraînement du modèle GP pour la composante principale ", i+1, "---")
        
        # Sélection de la iᵉ composante principale des données de sortie
        Y_train = tf.convert_to_tensor(y_train[:, i:i+1], dtype=tf.float64)
        
        # On utilise notre noyau "FunctionalL2Kernel" basé sur la distance L² entre fonctions.
        #kernel = FunctionalL2Kernel(lengthscale=param[0], variance=param[1]**2)  # ℓ : échelle de corrélation et σ² : variance du processus
        
        if kernel_fn is None:
            # noyau par défaut :
            kernel = FunctionalL2Kernel(lengthscale=param[0],variance=param[1]**2,n_entries=n_entries)
        elif isinstance(kernel_fn, gpflow.kernels.Kernel):
            # si un noyau déjà instancié est passé
            kernel = kernel_fn
        else:
            # kernel_fn est supposé être un constructeur → on l’appelle
            kernel = kernel_fn(param)
            
        # Modèle GP régressif
        model = gpflow.models.GPR(data=(X_train_flat, Y_train),kernel=kernel,mean_function=Constant()) # moyenne constante (apprise automatiquement)
        print("Modèle GP créé pour la composante principale ", i+1,". Optimisation des hyperparamètres...")
        # Optimisation des hyperparamètres
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))    
        print("Optimisation terminée.")
        models.append(model)
        if verbose:
            print("Les hyperparamètres optimisés sont :")
            gpflow.utilities.print_summary(model)
            print(f"-----Diagnostics pour la validation du modèle de la composante {i+1}-----")
            validation_GP(model.kernel.K(X_train_flat), Y_train)

    return models
    

def GP_predict(models, x_test, n_pc):
        print("Prédiction en cours...")
        means = []  # Liste des moyennes prédites pour chaque composante principale
        X_test_tf  = tf.convert_to_tensor(x_test, dtype=tf.float64)      # Conversion numpy -> tensorflow
        X_test_flat  = tf.reshape(X_test_tf,  (X_test_tf.shape[0],  -1)) # Reshape
        
        for i in range(n_pc): # Prédiction sur les nouvelles entrées
            mean_i, _ = models[i].predict_f(X_test_flat)
            means.append(mean_i.numpy().flatten())  # conversion TF → numpy
        mean = np.column_stack(means) # Chaque colonne correspond à la prédiction d’une composante principale
        return mean