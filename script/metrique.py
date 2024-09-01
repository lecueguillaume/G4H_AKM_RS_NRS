### SCRIPT POUR LES SIMULATIONS AUTOUR DE LA DENSITE, LE RMSE, ET L'AMI ###

# bibliothéques fondamentales
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random

# module maison
import script.decorateur as decorateur
import script.graph_formation as graph_formation
import script.MF_NRS as MF_NRS

GraphFormation = graph_formation.GraphFormation
get_estimations = MF_NRS.get_estimations

# Clustering
from sklearn.cluster import KMeans
from sklearn import metrics

# Fonctions qui calcul le root mean squared error entre les effets fixes; soit sur les deux, soit sur les effets fixes alpha/psi uniquement
def rmse(alpha_hat, psi_hat, alpha_star, psi_star):
    """
    Calcule l'erreur quadratique moyenne (RMSE) entre les effets fixes estimés et réels.

    Args:
        alpha_hat (array-like): Estimations des effets fixes alpha.
        psi_hat (array-like): Estimations des effets fixes psi.
        alpha_star (array-like): Vraies valeurs des effets fixes alpha.
        psi_star (array-like): Vraies valeurs des effets fixes psi.

    Returns:
        float: La racine carrée de l'erreur quadratique moyenne.
    """
    if not (len(alpha_hat) and len(psi_hat) and len(alpha_star) and len(psi_star)):
        raise ValueError("All input arrays must be non-empty")
    if not (len(alpha_hat) == len(alpha_star) and len(psi_hat) == len(psi_star)):
        raise ValueError("Corresponding input arrays must have the same length")
        
    S = 0
    for i in range(len(alpha_hat)):
        for j in range(len(psi_hat)):
            S += (alpha_hat[i] + psi_hat[j] - psi_star[j] - alpha_star[i])**2
    return np.sqrt(S/(len(alpha_hat)*len(psi_hat)))

def rmse_single(hat, star, effect_type):
    """
    Calcule l'erreur quadratique moyenne (RMSE) pour les effets fixes alpha ou psi.

    Args:
        hat (array-like): Estimations des effets fixes.
        star (array-like): Vraies valeurs des effets fixes.
        effect_type (str): 'alpha' ou 'psi' pour spécifier le type d'effet fixe.

    Returns:
        float: La racine carrée de l'erreur quadratique moyenne.
    """
    if not (len(hat) and len(star)):
        raise ValueError("Input arrays must be non-empty")
    if len(hat) != len(star):
        raise ValueError("Input arrays must have the same length")
    if effect_type not in ['alpha', 'psi']:
        raise ValueError("effect_type must be either 'alpha' or 'psi'")
    
    return np.sqrt(np.mean((np.array(hat) - np.array(star))**2))

def rmse_multiD(alpha_hat, psi_hat, alpha_star, psi_star):
    """
    Calcule l'erreur quadratique moyenne (RMSE) pour les effets fixes multidimensionnels.

    Args:
        alpha_hat (ndarray): Estimations des effets fixes alpha multidimensionnels.
        psi_hat (ndarray): Estimations des effets fixes psi multidimensionnels.
        alpha_star (ndarray): Vraies valeurs des effets fixes alpha multidimensionnels.
        psi_star (ndarray): Vraies valeurs des effets fixes psi multidimensionnels.

    Returns:
        float: La racine carrée de l'erreur quadratique moyenne.
    """
    
    if not all(arr.ndim == 2 for arr in [alpha_hat, psi_hat, alpha_star, psi_star]):
        raise ValueError("All input arrays must be 2-dimensional")
    if not (alpha_hat.shape == alpha_star.shape and psi_hat.shape == psi_star.shape):
        raise ValueError("Corresponding input arrays must have the same shape")
        
    S = 0
    for i in range(alpha_hat.shape[0]):
        for j in range(psi_hat.shape[0]):
            S += (np.dot(alpha_hat[i], psi_hat[j]) - np.dot(psi_star[j],alpha_star[i]))**2
    return np.sqrt(S/(psi_hat.shape[0]*alpha_hat.shape[0]))


def tab_ami_kmeans_inertia(graph_object, ef_patient, ef_doctor, save = False, return_ = False, k_max= 11):
    
    """
    Calcule et affiche les métriques de clustering (inertie et AMI) pour différentes valeurs de K
    en utilisant l'algorithme K-means sur les embedding factors des patients et des docteurs.

    Args:
        graph_object: Objet contenant les données du graphe.
        ef_patient (ndarray): Embedding factors des patients.
        ef_doctor (ndarray): Embedding factors des docteurs.
        save (bool): Si True, sauvegarde les résultats dans un fichier CSV. Par défaut False.
        return_ (bool): Si True, retourne le DataFrame des résultats. Par défaut False.

    Returns:
        pd.DataFrame ou None: DataFrame des résultats si return_=True, sinon None.
    """

    # Define the range of K values to test
    k_values = range(2, k_max)  # Start from 2 since K=1 doesn't make sense for clustering

    if ef_patient.shape[1] == 1:
        ef_patient = ef_patient.flatten().reshape(-1,1)
    if ef_doctor.shape[1] == 1:
        ef_doctor = ef_doctor.flatten().reshape(-1,1)
    
    # Initialize lists to store the metrics for each group
    inertia_p = []
    inertia_d = []
    ami_p = []
    ami_d = []
    
    # Loop through each K value
    for k in k_values:
        # KMeans for Group patients
        kmeans_p = KMeans(n_clusters=k, random_state=0)
        labels_p = kmeans_p.fit_predict(ef_patient)
        inertia_p.append(kmeans_p.inertia_)
        ami_p.append(metrics.adjusted_rand_score(graph_object.alpha_class, labels_p))
        
        # KMeans for doctors
        kmeans_d = KMeans(n_clusters=k, random_state=0)
        labels_d = kmeans_d.fit_predict(ef_doctor)
        inertia_d.append(kmeans_d.inertia_)
        ami_d.append(metrics.adjusted_rand_score(graph_object.psi_class, labels_d))
    
    # Create a DataFrame to hold the results
    results = pd.DataFrame({
        'K': k_values,
        'Inertia patient': inertia_p,
        'AMI patient': ami_p,
        'Inertia doctor': inertia_d,
        'AMI doctor': ami_d    })

    if save == True:
        results.to_csv('kmeans_metrics.csv', index=False)
    # Display the DataFrame
    print(results)

    if return_ == True:
        return results
    return None

