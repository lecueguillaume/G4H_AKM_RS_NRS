### SCRIPT POUR EFFECTUER DES SIMULATIONS POUR ETUDIER L'ESTIMATION DES BETA ###

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


@decorateur.log_execution_time('execution_details.txt')
def simulation_beta(n=100, nb_epochs=140, save=True):
    """
    Simule et estime les coefficients beta pour un graphe biparti, puis trace les distributions des estimations.

    Paramètres :
    ------------
    n : int, optionnel
        Le nombre de simulations à effectuer (par défaut 100).
    
    nb_epochs : int, optionnel
        Le nombre d'époques utilisées pour l'estimation dans chaque simulation (par défaut 140).
    
    save : bool, optionnel
        Si True, sauvegarde le graphique en tant qu'image PNG sous le nom 'Simulation_estimation_beta.png' (par défaut True).

    Retour :
    --------
    Aucun retour. La fonction génère et affiche un graphique. Si l'option `save` est True, elle sauvegarde également le graphique en tant qu'image PNG.
    """

    beta_X_p_hat, beta_X_d_hat, beta_D_p_hat, beta_D_d_hat, beta_distance_hat = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n) 
    for k in range(n):
        graph_object= GraphFormation(
                        n_patients=1000,
                         n_doctors=50,
                         max_number_connections=50,
                        beta_distance_graph = -25,
                        seed= k+1)
        graph_object.do_the_graph()

        estimates = get_estimations(graph_object.df, nb_epochs=nb_epochs)
        beta_X_p_hat[k] = estimates[1][2][0]
        beta_X_d_hat[k] = estimates[1][3][0]
        beta_D_p_hat[k] =  estimates[1][4][0]
        beta_D_d_hat[k] =  estimates[1][5][0]
        beta_distance_hat[k] =  estimates[1][6][0]
        print(f"Simulation {k+1}: done !")
    # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    title = ["simulation beta X patient", "simulation beta X docteur", "simulation beta D patient", "simulation beta D docteur", "simulation beta distance"]
    beta = [ beta_X_p_hat,  beta_X_d_hat,  beta_D_p_hat,  beta_D_d_hat,  beta_distance_hat]
    true_beta = [graph_object.beta_X_p_graph, graph_object.beta_X_d_graph, graph_object.beta_D_p_graph, graph_object.beta_D_d_graph, graph_object.beta_distance_graph]
    for k in range(5):
        axs[k].set_title(title[k])
        axs[k].hist(beta[k])
        
        # Ajouter un texte pour indiquer la moyenne
        mean_value = np.mean(beta[k])
        axs[k].axvline(mean_value, color='red', linestyle='dashed', linewidth=2)
        axs[k].text(mean_value, max(np.histogram(beta[k], bins=20)[0]), f'Moyenne: {mean_value:.2f}', color='red')

        # Ajouter la vrai valeur
        axs[k].axvline(true_beta[k], color='green', linestyle='dashed', linewidth=2)
        axs[k].text(true_beta[k], max(np.histogram(beta[k], bins=20)[0]), f'Vrai beta: {true_beta[k]:.2f}', color='green')
    
    # sauve le plot en png
    if save == True:
        plt.savefig('Simulation_estimation_beta.png')
    plt.tight_layout()
    plt.show()


