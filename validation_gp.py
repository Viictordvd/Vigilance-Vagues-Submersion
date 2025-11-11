import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def loo_tf(K, y_true):
    # K : matrice de covariance, tf.Tensor (n,n); y_true : valeurs réelles de sorties, tf.Tensor (n,)
    L = tf.linalg.cholesky(K)  # Matrice triangulaire inférieure
    # Calcul de alpha = K^{-1} y
    alpha = tf.linalg.cholesky_solve(L, y_true)
    alpha = tf.squeeze(alpha, -1)

    # Calcul de l'inverse de L
    n = tf.shape(K)[0]
    I = tf.eye(n, dtype=K.dtype)
    invL = tf.linalg.triangular_solve(L, I)  # shape (n,n)

    # Q_diag = sum_k invL[k,i]^2 = sum over rows of invL**2 along axis=0
    Q_diag = tf.reduce_sum(invL**2, axis=0)  # shape (n,)
    #Moyenne et variance loo
    sigma2_loo = 1.0 / Q_diag
    mu_loo = tf.squeeze(y_true) - alpha / Q_diag
    return mu_loo, sigma2_loo

# Fonction de validation du modèle
def validation_GP(K,y_true):
    #model : modèle de GP gpflow.models.GPR , y_true: valeurs réelles de sorties, tf.Tensor (n,)
    mu_loo, sigma2_loo = loo_tf(K,y_true)
    y_true = np.asarray(y_true).flatten()
    mu_loo = np.asarray(mu_loo).flatten()
    sigma2_loo = np.asarray(sigma2_loo).flatten()

    # Calcul des résidus et des résidus standardisés
    standardized_residuals = (y_true - mu_loo) / np.sqrt(sigma2_loo)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1 : Prédictions LOO vs Valeurs réelles ---
    ax = axes[0]
    ax.scatter(y_true, mu_loo, alpha=0.7, color='steelblue', edgecolor='k')
    lims = [min(y_true.min(), mu_loo.min()), max(y_true.max(), mu_loo.max())]
    ax.plot(lims, lims, 'r--', lw=2, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Prédictions LOO")
    ax.set_title("Prédictions LOO vs Valeurs réelles")
    ax.legend()
    ax.grid(True, linestyle=':')

    # --- Plot 2 : Résidus standardisés ---
    ax = axes[1]
    ax.axhline(0, color='k', lw=1)
    ax.axhline(2, color='r', ls='--')
    ax.axhline(-2, color='r', ls='--')
    ax.scatter(np.arange(len(standardized_residuals)),
               standardized_residuals, color='steelblue', edgecolor='k')
    ax.set_xlabel("Index de l'observation")
    ax.set_ylabel("Résidu standardisé")
    ax.set_title("Résidus standardisés LOO")
    ax.grid(True, linestyle=':')

    # --- Plot 3 : QQ-plot des résidus standardisés ---
    ax = axes[2]
    stats.probplot(standardized_residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_color('steelblue')  # points
    ax.get_lines()[1].set_color('r')          # droite de référence
    ax.set_title("QQ-plot des résidus standardisés")

    plt.tight_layout()
    plt.show()