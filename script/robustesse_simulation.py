### SCRIPT POUR LES SIMULATIONS POUR TESTER LA ROBUSTESSE ###

# bibliothéques fondamentales
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random

# module maison
import script.decorateur as decorateur
import script.graph_formation as graph_formation
import script.MF_HNS as MF_HNS


get_estimations = MF_HNS.get_estimations

@decorateur.compute_time
def robustesse_simul_n_models(n, train_ratio, graph_object, nb_epochs = 80, regularization="l2", l_lambda = 1e-7):

    """
    Effectue des simulations pour évaluer la robustesse des estimations de paramètres d'un modèle.

    Paramètres :
    -----------
    n : int
        Le nombre de simulations à effectuer.
    train_ratio : float
        Le ratio de division entre les données d'entraînement et de test.
    graph_object : object
        Un objet contenant les données et les paramètres du graphe.
    nb_epochs : int, optionnel
        Le nombre d'époques pour l'entraînement du modèle. Par défaut à 80.
    regularization : str, optionnel
        Le type de régularisation à utiliser. Par défaut "l2".
    l_lambda : float, optionnel
        Le paramètre de régularisation lambda. Par défaut à 1e-7.

    Retourne :
    --------
    matrix_ef_patient : numpy.ndarray
        Une matrice contenant les estimations pour les patients à chaque simulation.
    matrix_ef_doctor : numpy.ndarray
        Une matrice contenant les estimations pour les docteurs à chaque simulation.
    beta_X_p : numpy.ndarray
        Les valeurs de beta_X pour les patients à chaque simulation.
    beta_X_d : numpy.ndarray
        Les valeurs de beta_X pour les docteurs à chaque simulation.
    beta_D_p : numpy.ndarray
        Les valeurs de beta_D pour les patients à chaque simulation.
    beta_D_d : numpy.ndarray
        Les valeurs de beta_D pour les docteurs à chaque simulation.
    beta_distance : numpy.ndarray
        Les valeurs de la distance beta à chaque simulation.

    Description :
    ------------
    Cette fonction effectue `n` simulations en ré-échantillonnant aléatoirement les données et en séparant les données en ensembles d'entraînement et de test
    selon le ratio spécifié. Pour chaque simulation, elle entraîne un modèle et stocke les estimations des paramètres dans des matrices. Ensuite, elle affiche
    des histogrammes des estimations des paramètres avec des lignes indiquant les valeurs moyennes et les vraies valeurs des paramètres.

    Exemple :
    --------
    graph_obj = GraphObject(...)
    matrix_ef_patient, matrix_ef_doctor, beta_X_p, beta_X_d, beta_D_p, beta_D_d, beta_distance = robustesse_simul_n_models(100, 0.8, graph_obj)
    """
    
    # On va découper le df en 2 à partir de l'indice "train_indices"
    df = graph_object.df
    train_indices = int(df.shape[0]*train_ratio)
    
    matrix_ef_patient, matrix_ef_doctor = np.zeros((n, graph_object.n_patients)), np.zeros((n,graph_object.n_doctors))

    beta_X_p, beta_X_d, beta_D_p, beta_D_d, beta_distance = np.zeros(n,), np.zeros(n,), np.zeros(n,), np.zeros(n,), np.zeros(n,)
    
    for k in range(n):
        df_shuffled = df.sample(frac=1, random_state=k+1).reset_index(drop=True)
        df_train, df_test = df_shuffled.iloc[:train_indices,:] , df_shuffled.iloc[train_indices:,:]
        
        estimates = get_estimations(df_train, seed = k+1, show_print=0, nb_epochs= nb_epochs, regularization= regularization, l_lambda = l_lambda)
        
        matrix_ef_patient[k][:] = estimates[1][0].flatten()
        matrix_ef_doctor[k][:] = estimates[1][1].flatten()
        beta_X_p[k] = estimates[1][2]
        beta_X_d[k] = estimates[1][3]
        beta_D_p[k] = estimates[1][4]
        beta_D_d[k] = estimates[1][5]
        beta_distance[k] = estimates[1][6]
        print(f"simulation {k+1}: done")
    
        # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    title = ["simulation beta X patient", "simulation beta X docteur", "simulation beta D patient", "simulation beta D docteur", "simulation beta distance"]
    beta = [beta_X_p, beta_X_d, beta_D_p, beta_D_d, beta_distance]
    true_values = [graph_object.beta_X_p_graph, graph_object.beta_X_d_graph, graph_object.beta_D_p_graph, graph_object.beta_D_d_graph, graph_object.beta_distance_graph]
    for k in range(5):
        axs[k].hist(beta[k], bins=25)
        axs[k].set_title(title[k])
        
        # Ajouter un texte pour indiquer la moyenne
        mean_value = np.mean(beta[k])
        axs[k].axvline(mean_value, color='red', linestyle='dashed', linewidth=2)
        axs[k].text(mean_value, max(np.histogram(beta[k], bins=25)[0]), f'Moyenne: {mean_value:.2f}', color='red')

        axs[k].axvline(true_values[k], color='green', linestyle='dashed', linewidth=2)
        axs[k].text(true_values[k], max(np.histogram(beta[k], bins=25)[0]), f'vrai valeur: {true_values[k]:.2f}', color='green')
        
    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    plt.show()

    return matrix_ef_patient, matrix_ef_doctor, beta_X_p, beta_X_d, beta_D_p, beta_D_d, beta_distance       


def make_distance_ef_matrix(ef):
    size = ef.shape[0]
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i):
            matrix[i][j] = np.abs(ef[i] - ef[j])
    return matrix

def make_list_dist_ef_matrix(ef_simulation, triangle=False):
    l = []
    for k in range(ef_simulation.shape[0]):
        ef = ef_simulation[k]
        A_k = make_distance_ef_matrix(ef)
        if triangle==True:
            l.append(A_k)
        else:
            l.append(A_k + np.transpose(A_k))
    return l

def compute_cosine_similarity(distance_matrix_list_ef, triangle=True):
    size = len(distance_matrix_list_ef)
    cosine_sim = np.zeros((size,size))
    for i in range(size):
        for j in range(i):
            A_flat = distance_matrix_list_ef[i].flatten().reshape(1, -1)
            B_flat = distance_matrix_list_ef[j].flatten().reshape(1, -1)
            cosine_sim[i][j] = cosine_similarity(A_flat,B_flat)[0][0]
    if triangle == True:
        return cosine_sim + np.transpose(cosine_sim)
    else:
        return cosine_sim



#%%capture captured_output
#results = robustesse_simulation.robustesse_simul_n_models(100, 0.85, graph_object)
# Enregistrer le tuple sur le disque
#with open('results_simulation_85.pkl', 'wb') as f:
    #pickle.dump(results, f)

# Charger la variable depuis le disque
#with open('results_simulation_85.pkl', 'rb') as f:
#    results_simulation = pickle.load(f)

#Pour tester la robustesse des effets fixes, nous allons construire des matrices $A_{k}$ dont les composantes $(a_{ij}^{(k)})$ valent #$|\alpha_{i}^{(k)} - \alpha_{j}^{(k)}|$   (ou $d(\alpha_{i}^{(k)},\alpha_{j}^{(k)})$ de façon plus général). Nous allons ensuite comparer la similarité entre ces matrices $A_{k}$.

#distance_matrix_list_ef_patient_triangle = robustesse_simulation.make_list_dist_ef_matrix(results_simulation[0])
#distance_matrix_list_ef_docteur_triangle = robustesse_simulation.make_list_dist_ef_matrix(results_simulation[1])