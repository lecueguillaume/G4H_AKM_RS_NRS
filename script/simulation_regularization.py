### SCRIPT SIMULATION POUR DETERMINER LE LAMBDA DE REGULARIZATION OPTIMAL ### 

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
import script.density_ami_rmse_simulation as density_ami_rmse_simulation

GraphFormation = graph_formation.GraphFormation
get_estimations = MF_HNS.get_estimations

@decorateur.log_execution_time('execution_details.txt')
def plot_rmse_lambda( regu_lambdas = np.array([10**(-i) for i in range(5,11)]), nb_epochs=200, regularization="l2", save=True ):
    
    size = len(regu_lambdas)
    rmse_array, rmse_alpha_array, rmse_psi_array = np.zeros(size), np.zeros(size), np.zeros(size)
    
    for i,regu_lambda in enumerate(regu_lambdas):
        graph_object= GraphFormation(
                        n_patients=1000,
                         n_doctors=50,
                         max_number_connections=50,
                        beta_distance_graph = -25,)
        graph_object.do_the_graph()

        estimates =  get_estimations(graph_object.df, nb_epochs=nb_epochs, seed=12, l_lambda = regu_lambda, regularization="l2")
        
        alpha_hat = np.array(estimates[1][0])
        psi_hat = np.array(estimates[1][1])

        rmse_array[i] =  density_ami_rmse_simulation.rmse(alpha_hat, psi_hat, alpha_star = graph_object.alpha_graph, psi_star = graph_object.psi_graph)[0]
        rmse_alpha_array[i] =  density_ami_rmse_simulation.rmse_alpha(alpha_hat, graph_object.alpha_graph)[0]
        rmse_psi_array[i] =  density_ami_rmse_simulation.rmse_psi(psi_hat, graph_object.psi_graph)[0]

        print(f"Simulation {i+1}: done !")

    x = -np.log10(regu_lambdas)
    # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))
    
    axs[0].set_title(f"Rmse en fonction du lambda (10^(-i) pour l'échelle) de régularization " + regularization)
    axs[0].plot(x, rmse_array)
    axs[0].set_xlabel('lambda (10^(-x))')
    axs[0].set_ylabel('RMSE')

    axs[0].set_title(f"Rmse des alpha et des psi en fonction du lambda (10^(-i) pour l'échelle) de régularization " + regularization)
    axs[1].plot(x, rmse_alpha_array, label="RMSE of psi")
    axs[1].plot(x, rmse_psi_array, label="RMSE of alpha")
    axs[1].set_xlabel('lambda (10^(-x))')
    axs[1].set_ylabel('RMSE')
    axs[1].legend()
    
    if save == True:
        plt.savefig('Simulation_regularization_lambda.png')
    plt.tight_layout()
    plt.show()    