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
    S = 0
    for i in range(len(alpha_hat)):
        for j in range(len(psi_hat)):
            S += (alpha_hat[i] + psi_hat[j] - psi_star[j] - alpha_star[i])**2
    return np.sqrt(S/(len(alpha_hat)*len(psi_hat)))
    
def rmse_psi(psi_hat, psi_star):
    S = 0
    for j in range(len(psi_hat)):
        S += (psi_hat[j] - psi_star[j])**2
        return np.sqrt(S/len(psi_hat))

def rmse_alpha(alpha_hat, alpha_star):
    S = 0
    for j in range(len(alpha_hat)):
        S += (alpha_hat[j] - alpha_star[j])**2
        return np.sqrt(S/len(alpha_hat))

@decorateur.compute_time
def plot_rmse_epochs(sim_beta_distance_array = [-25,-20, -17, -15,-12,-10, -8, -7,], epochs_step = 10, n=10, save=True):

    """
    Trace l'évolution de l'erreur quadratique moyenne (RMSE) des effets fixes à travers les époques pour différentes valeurs de densité (définies par sim_beta_distance).

    Paramètres :
    ------------
    sim_beta_distance_array : list, optionnel
        Une liste de valeurs de sim_beta_distance (par défaut [-20, -15, -12, -10, -7, -5, -3, -2]).
        Chaque valeur représente une densité différente pour la formation du graphe.
    
    epochs_step : int, optionnel
        Le nombre d'époques entre chaque étape de mesure du RMSE (par défaut 10).
    
    n : int, optionnel
        Le nombre de mesures de RMSE à effectuer (par défaut 10).
    
    save : bool, optionnel
        Si True, sauvegarde le graphique en tant qu'image PNG sous le nom 'RMSE_vs_epochs_density.png' (par défaut False).

    Retour :
    --------
    Aucun retour. La fonction génère et affiche un graphique. Si l'option `save` est True, elle sauvegarde également le graphique en tant qu'image PNG.
    """
    
    for sim_beta_distance in sim_beta_distance_array:
        
        rmse_array = np.zeros(n)
        graph_object= GraphFormation(
                        n_patients=1000,
                         n_doctors=50,
                         max_number_connections=50,
                        beta_distance_graph = sim_beta_distance)
        graph_object.do_the_graph()
        
        for i,step in enumerate(np.arange(epochs_step, epochs_step*(n+1), epochs_step)):
            if i==0:
                estimates = get_estimations( graph_object.df, nb_epochs=epochs_step, show_print=0,  initial_weights=None)
            else:
                estimates = get_estimations( graph_object.df, nb_epochs=epochs_step, show_print=0,  initial_weights=estimates[1])
            alpha_hat = estimates[1][0]
            psi_hat = estimates[1][1]
            rmse_array[i] = rmse(alpha_hat = alpha_hat, psi_hat = psi_hat, alpha_star = graph_object.alpha_graph, psi_star = graph_object.psi_graph)[0]
        plt.plot(np.arange(epochs_step, epochs_step*(n+1), epochs_step), rmse_array, label=f"beta distance = {sim_beta_distance}/ density = {graph_object.density*100:.2f}%")

    plt.title("évolution du RMSE des effets fixes à travers \n les epoques pour différentes densités")
    plt.xlabel("epochs")
    plt.ylabel('RMSE')
    plt.legend()
    if save==True:
        # Sauvegarder le graphique en tant qu'image PNG
        plt.savefig('RMSE_vs_epochs_density.png')
   
    plt.show()      

@decorateur.compute_time
def plot_rmsedensity( sim_beta_distance_array = [-25, -20,-15, -12, -10,-8,-5, -3,], nb_epochs = 200):
    size = len(sim_beta_distance_array)
    density_array = np.zeros(size)
    beta_skilled_d, beta_age_d, beta_informed_p, beta_age_p, beta_distance = np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size) ,np.zeros(size)
    rmse_array, rmse_alpha_array, rmse_psi_array = np.zeros(size), np.zeros(size), np.zeros(size)
    ami_p_array, ami_d_array =  np.zeros(size), np.zeros(size)

    i=0
    for sim_beta_distance in sim_beta_distance_array:
        graph_all= graph_formation(
                    n_patients=1000,
                     n_doctors=50,
                     max_number_connections=50,
                     z=1.42,
                    beta_distance_graph = sim_beta_distance,
                    seed=i)
       
        df, alpha_star, psi_star, alpha_class, psi_class, coor_patients, coor_doctors, D, beta_age_p_star, beta_age_d_star, beta_informed_p_star,\
        beta_skilled_d_star, beta_distance_star, sim_patient_age_normed, sim_doctor_age_normed, sim_patient_informed, sim_doctor_skilled = graph_all
        
        density_array[i] = density(df)
        print(f"density: {100*density_array[i]:2f}%")
        
        estimates =  get_estimations(df, nb_epochs=80, initial_weights=None, target_loss=None, l_lambda=0, show_print=0, seed=12)
        
        ef_patient_hat = np.array(estimates[1][0])
        ef_doctor_hat = np.array(estimates[1][1])

        rmse_array[i] = rmse(ef_patient_hat, ef_doctor_hat, alpha_star, psi_star)[0]
        rmse_alpha_array[i] = rmse_alpha(ef_patient_hat, alpha_star)[0]
        rmse_psi_array[i] = rmse_psi(ef_doctor_hat, psi_star)[0]

        # Initialise le modèle KMeans avec 3 et 2 clusters
        kmeans_p = KMeans(n_clusters=3, random_state=i)
        kmeans_d = KMeans(n_clusters=3, random_state=i)
        
        ef_patient_hat = np.array(ef_patient_hat).flatten()
        ef_doctor_hat = np.array(ef_doctor_hat).flatten()
        
        # Ajustement des Kmeans sur les ef estimés
        kmeans_p.fit(ef_patient_hat.reshape(-1,1))
        kmeans_d.fit(ef_doctor_hat.reshape(-1,1))
       
        # Ajouter les labels prédits au DataFrame
        cluster_patient = kmeans_p.labels_
        cluster_doctor = kmeans_d.labels_

        ami_p_array[i], ami_d_array[i]  = metrics.adjusted_rand_score(alpha_class, cluster_patient), metrics.adjusted_rand_score(psi_class, cluster_doctor)
        i+=1

    # Créer une figure avec deux sous-graphiques (subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Trace le premier graphique
    ax1.plot(density_array, rmse_array, label="Global RMSE")
    ax1.plot(density_array, rmse_alpha_array, label="RMSE of psi")
    ax1.plot(density_array, rmse_psi_array, label="RMSE of alpha")
    ax1.set_title("RMSE du graphe en fonction de sa densité")
    ax1.set_xlabel('density')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    # Trace le deuxième graphique
    ax2.plot(density_array, ami_p_array, label="AMI pour les patients")
    ax2.plot(density_array, ami_d_array, label="AMI pour les doctors")
    ax2.set_title('AMI des ef estimés en fonction de la densité')
    ax2.set_xlabel('density')
    ax2.set_ylabel('AMI')
    ax2.legend()
    # Ajuste l'espace entre les sous-graphiques
    plt.tight_layout()
    plt.show()    
    
    return density_array, ami_p_array, ami_d_array, beta_age_p, beta_age_d, beta_informed_p, beta_skilled_d, beta_distance

@decorateur.log_execution_time('execution_details.txt')
def simulation_density_ami(n=10, sim_beta_distance_array = [-25, -20,-15, -12, -10,-8,-5, -3,], nb_epochs=200, n_patients=1000, n_doctors=50, n_clusters_p = 3, n_clusters_d=3  ):
    
    size = len(sim_beta_distance_array)
    matrix = np.zeros((n,size,4))

    for j in range(n):
        density_array = np.zeros(size)
        rmse_array, rmse_alpha_array, rmse_psi_array = np.zeros(size), np.zeros(size), np.zeros(size)
        ami_p_array, ami_d_array =  np.zeros(size), np.zeros(size)
    
        for i,sim_beta_distance in enumerate(sim_beta_distance_array):
            
            graph_object= GraphFormation(
                        n_patients=n_patients,
                         n_doctors=n_doctors,
                        beta_distance_graph = sim_beta_distance)
            graph_object.do_the_graph()
            
            density_array[i] = graph_object.density
            
            estimates =  get_estimations(graph_object.df, nb_epochs=nb_epochs, initial_weights=None, target_loss=None, l_lambda=0, show_print=0, seed=j)
            
            ef_patient_hat = np.array(estimates[1][0])
            ef_doctor_hat = np.array(estimates[1][1]) 
    
            rmse_array[i] = rmse(ef_patient_hat, ef_doctor_hat, graph_object.alpha_graph, graph_object.psi_graph)[0]
            #rmse_alpha_array[i] = rmse_alpha(ef_patient_hat, alpha_star)[0]
            #rmse_psi_array[i] = rmse_psi(ef_doctor_hat, psi_star)[0]
    
            # Initialise le modèle KMeans avec 3 et 2 clusters
            kmeans_p = KMeans(n_clusters = n_clusters_p, random_state=j)
            kmeans_d = KMeans(n_clusters = n_clusters_d, random_state=j)
            
            ef_patient_hat = np.array(ef_patient_hat).flatten()
            ef_doctor_hat = np.array(ef_doctor_hat).flatten()
            
            # Ajustement des Kmeans sur les ef estimés
            kmeans_p.fit(ef_patient_hat.reshape(-1,1))
            kmeans_d.fit(ef_doctor_hat.reshape(-1,1))
           
            # Ajouter les labels prédits au DataFrame
            cluster_patient = kmeans_p.labels_
            cluster_doctor = kmeans_d.labels_
    
            ami_p_array[i], ami_d_array[i]  = metrics.adjusted_rand_score(graph_object.alpha_class, cluster_patient), metrics.adjusted_rand_score(graph_object.psi_class, cluster_doctor)

            matrix[j][i][0] = density_array[i]
            matrix[j][i][1] = rmse_array[i]
            matrix[j][i][2] = ami_p_array[i]
            matrix[j][i][3] = ami_d_array[i]

        print(f"Simulation {j+1}: done!")

    return matrix

@decorateur.log_execution_time('execution_details.txt')
def simulation_ami_dilatation(n=10, patient_dilatation = True, doctor_dilatation = True, sim_dilatation_array = [0.2, 0.5, 1, 2, 3, 4, 5, 6], sim_beta_distance_array = [-5,-8, -10, -12,-15,-20, -25,-30],  nb_epochs=200, n_patients=1000, n_doctors=50 ):

    assert len(sim_dilatation_array) == len(sim_beta_distance_array)
    size = len(sim_beta_distance_array)
    matrix = np.zeros((n,size,4))

    for j in range(n):
        density_array = np.zeros(size)
        rmse_array, rmse_alpha_array, rmse_psi_array = np.zeros(size), np.zeros(size), np.zeros(size)
        ami_p_array, ami_d_array =  np.zeros(size), np.zeros(size)
    
        for i,sim_dilatation in enumerate(sim_dilatation_array):

            if (patient_dilatation == True) and (doctor_dilatation == False):
                
                graph_object= GraphFormation(
                            n_patients=n_patients,
                             n_doctors= n_doctors,
                            beta_distance_graph = sim_beta_distance_array[i],
                            dilatation_p = sim_dilatation,
                )
            elif (patient_dilatation == False) and (doctor_dilatation == True):
                graph_object= GraphFormation(
                            n_patients=n_patients,
                             n_doctors= n_doctors,
                            beta_distance_graph = sim_beta_distance_array[i],
                            dilatation_d = sim_dilatation
                )
            else:
                graph_object= GraphFormation(
                            n_patients=n_patients,
                            n_doctors= n_doctors,
                            beta_distance_graph = sim_beta_distance_array[i],
                            dilatation_d = sim_dilatation,
                            dilatation_p = sim_dilatation
                )
                
            graph_object.do_the_graph()
            
            density_array[i] = graph_object.density
        
            estimates =  get_estimations(graph_object.df, nb_epochs=nb_epochs, initial_weights=None, target_loss=None, l_lambda=0, show_print=0, seed=j)
            
            ef_patient_hat = np.array(estimates[1][0])
            ef_doctor_hat = np.array(estimates[1][1]) 
    
            rmse_array[i] = rmse(ef_patient_hat, ef_doctor_hat, graph_object.alpha_graph, graph_object.psi_graph)[0]
            #rmse_alpha_array[i] = rmse_alpha(ef_patient_hat, alpha_star)[0]
            #rmse_psi_array[i] = rmse_psi(ef_doctor_hat, psi_star)[0]
    
            # Initialise le modèle KMeans avec 3 et 2 clusters
            kmeans_p = KMeans(n_clusters=3, random_state=j)
            kmeans_d = KMeans(n_clusters=3, random_state=j)
            
            ef_patient_hat = np.array(ef_patient_hat).flatten()
            ef_doctor_hat = np.array(ef_doctor_hat).flatten()
            
            # Ajustement des Kmeans sur les ef estimés
            kmeans_p.fit(ef_patient_hat.reshape(-1,1))
            kmeans_d.fit(ef_doctor_hat.reshape(-1,1))
           
            # Ajouter les labels prédits au DataFrame
            cluster_patient = kmeans_p.labels_
            cluster_doctor = kmeans_d.labels_
    
            ami_p_array[i], ami_d_array[i]  = metrics.adjusted_rand_score(graph_object.alpha_class, cluster_patient), metrics.adjusted_rand_score(graph_object.psi_class, cluster_doctor)

            matrix[j][i][0] = density_array[i]
            matrix[j][i][1] = rmse_array[i]
            matrix[j][i][2] = ami_p_array[i]
            matrix[j][i][3] = ami_d_array[i]
            

    return matrix


def plot_simulation_density_ami(matrix, rank=7, n=100, worst=1, save=True):
    mean_density_array,  worst_ami_p, worst_ami_d, mean_rmse_array  = np.zeros(rank), np.zeros(rank), np.zeros(rank),  np.zeros(rank)

    # Crée une figure avec deux sous-graphiques (subplots)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 12))
    fig.subplots_adjust(hspace=2, wspace=1)  # Ajuster l'espacement vertical et horizontal
    
    for j in range(rank):
        density_array, ami_p_array, ami_d_array, rmse_array = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        
        for i in range(n):
            density_array[i], rmse_array[i], ami_p_array[i], ami_d_array[i] = matrix[i][j][0], matrix[i][j][1], matrix[i][j][2], matrix[i][j][3]
        
        mean_density_array[j] = density_array.mean()   # récupère la moyenne des densités sur l'ensemble des simulations pour chaque beta distance différent
        mean_rmse_array[j] = rmse_array.mean()
        worst_ami_p[j] = np.sort(ami_p_array)[worst-1]     # récupére la 5eme pire AMI pour un niveau de densité
        worst_ami_d[j] = np.sort(ami_d_array)[worst-1]
        
    for i in range(n):
        ami_p_plot, ami_d_plot, density_plot, rmse_plot = np.zeros(rank), np.zeros(rank), np.zeros(rank),  np.zeros(rank)
        for j in range(rank):
            density_plot[j], rmse_plot[j], ami_p_plot[j], ami_d_plot[j] = matrix[i][j][0], matrix[i][j][1], matrix[i][j][2], matrix[i][j][3]
            
        ax1.plot(mean_density_array, ami_p_plot, c="blue", alpha = 3/n )
        ax2.plot(mean_density_array, ami_d_plot, c="orange", alpha = 3/n )
        ax3.plot(mean_density_array, rmse_plot, alpha = 3/n)

    ax1.plot(mean_density_array, worst_ami_p, c="red", label= f"{worst}th/st worst")
    ax1.set_ylabel('AMI')
    ax1.set_xlabel('density')
    ax1.set_ylim(0,1)
    ax1.legend()
    ax2.plot(mean_density_array, worst_ami_d, c="red", label= f"{worst}th/st worst")
    ax2.set_ylabel('AMI')
    ax2.set_xlabel('density')
    ax2.set_ylim(0,1)
    ax2.legend()
    ax3.plot(mean_density_array, mean_rmse_array, c='red', label=f'mean of the {n} simulation')
    ax3.set_ylabel("RMSE")
    ax3.set_xlabel('density')
    ax3.legend()

    ax4.plot(mean_density_array, mean_rmse_array)
    ax5.plot(mean_density_array, worst_ami_p, c="blue", label="patient")
    ax5.plot(mean_density_array, worst_ami_d, c="orange", label = "doctor")
    ax5.set_ylim(0,1)
    ax5.legend()

    if save == True:
        # Enregistrement du graphique dans un fichier
        fig.savefig('density_am.png', dpi=300, bbox_inches='tight')

    ax1.set_title(f'AMI des patients des {n} simulations en fonction de la densité')
    ax2.set_title(f'AMI des docteurs (orange) des {n} simulations en fonction de la densité')
    ax3.set_title(f'RMSE des {n} en fonction de la densité')
    ax4.set_title(f'Moyenne des RMSE sur les {n} simulations en fonction de la densité')
    ax5.set_title(f'{worst}eme/er plus mauvais AMI des patients (bleu) et des docteurs (orange) en fonction de la densité')

    
    plt.show()


def plot_simulation_dilatation_ami(matrix, dilatation_array, n=10, x_label = "coefficient de dilatation", worst=1, save = True):

    rank = len(dilatation_array)
    mean_density_array,  worst_ami_p, worst_ami_d, mean_rmse_array  = np.zeros(rank), np.zeros(rank), np.zeros(rank),  np.zeros(rank)

    # Crée une figure avec deux sous-graphiques (subplots)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 12))
    fig.subplots_adjust(hspace=2, wspace=1)  # Ajuster l'espacement vertical et horizontal
    
    for j in range(rank):
        density_array, ami_p_array, ami_d_array, rmse_array = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        
        for i in range(n):
            density_array[i], rmse_array[i], ami_p_array[i], ami_d_array[i] = matrix[i][j][0], matrix[i][j][1], matrix[i][j][2], matrix[i][j][3]
        
        mean_density_array[j] = density_array.mean()   # récupère la moyenne des densités sur l'ensemble des simulations pour chaque beta distance différent
        mean_rmse_array[j] = rmse_array.mean()
        worst_ami_p[j] = np.sort(ami_p_array)[worst-1]     # récupére la 5eme pire AMI pour un niveau de densité
        worst_ami_d[j] = np.sort(ami_d_array)[worst-1]
        
    for i in range(n):
        ami_p_plot, ami_d_plot, density_plot, rmse_plot = np.zeros(rank), np.zeros(rank), np.zeros(rank),  np.zeros(rank)
        for j in range(rank):
            density_plot[j], rmse_plot[j], ami_p_plot[j], ami_d_plot[j] = matrix[i][j][0], matrix[i][j][1], matrix[i][j][2], matrix[i][j][3]
            
        ax1.plot(dilatation_array, ami_p_plot, c="blue", alpha = 3/n )
        ax2.plot(dilatation_array, ami_d_plot, c="orange", alpha = 3/n )
        ax3.plot(dilatation_array, rmse_plot, alpha = 3/n)

    ax1.plot(dilatation_array, worst_ami_p, c="red", label= f"{worst}th/st worst")
    ax1.set_ylabel('AMI')
    ax1.set_xlabel(x_label)
    ax1.set_ylim(0,1)
    ax1.legend()
    ax2.plot(dilatation_array, worst_ami_d, c="red", label= f"{worst}th/st worst")
    ax2.set_ylabel('AMI')
    ax2.set_xlabel(x_label)
    ax2.set_ylim(0,1)
    ax2.legend()
    ax3.plot(dilatation_array, mean_rmse_array, c='red', label=f'mean of the {n} simulation')
    ax3.set_ylabel("RMSE")
    ax3.set_xlabel(x_label)
    ax3.legend()

 
    ax4.plot(dilatation_array, worst_ami_p, c="blue", label="patient")
    ax4.plot(dilatation_array, worst_ami_d, c="orange", label='doctor')
    ax4.set_ylim(0,1)
    ax4.set_ylabel("AMI")
    ax4.set_xlabel(x_label)
    ax4.legend()
    
    ax1.set_title(f'AMI des patients des {n} simulations en fonction du ' + x_label)
    ax2.set_title(f'AMI des docteurs (orange) des {n} simulations en fonction du  ' + x_label)
    ax3.set_title(f'RMSE des {n} en fonction du ' + x_label)
    ax4.set_title(f'{worst}eme/er plus mauvais AMI des patients (bleu) et des docteurs (orange) en fonction du ' + x_label)

    ax5.plot(dilatation_array, mean_density_array)
    ax5.set_title(f'density des graphes en fonction du ' + x_label)
    ax5.set_ylabel("density")
    ax5.set_xlabel(x_label)

    if save == True:
        # Enregistrement du graphique dans un fichier
        fig.savefig('dilatation_ami.png', dpi=300, bbox_inches='tight')
         
    plt.show()