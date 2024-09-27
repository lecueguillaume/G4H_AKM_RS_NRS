# bibliothéques fondamentales
import matplotlib.pyplot as plt
import pandas as pd
#import bipartitepandas as bpd
import numpy as np
import time
import random

import script.decorateur as decorateur
import script.graph_formation as graph_formation

# Pour visualisation/ réduction de dimension
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

import seaborn as sns
# Clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics

# modules maison
import script.decorateur as decorateur
import script.graph_formation as graph_formation
import script.MF_NRS as MF_NRS
import script.metrique as metrique

GraphFormation = graph_formation.GraphFormation
get_estimations = MF_NRS.get_estimations


def plot_heatmap(matrix, who = "patient", title = "Heatmap des AMI", xticklabel = np.array([1,2,3,4,5]), yticklabel = np.array([1, 2, 3, 4, 5, 8, 10]), save = False):
    """
    Plot a heatmap for AMI (Adjusted Mutual Information) data.

    Parameters:
    - matrix (numpy.ndarray): 3D array containing AMI data for patients and doctors.
    - who (str): Specifies whether to plot for 'patient' or 'doctor'. Default is 'patient'.
    - title (str): Title of the heatmap. Default is "Heatmap des AMI".
    - xticklabel (numpy.ndarray): Labels for x-axis. Default is [1,2,3,4,5].
    - yticklabel (numpy.ndarray): Labels for y-axis. Default is [1,2,3,4,5,8,10].
    - save (bool): If True, saves the plot as an image file. Default is False.

    Raises:
    - ValueError: If 'who' is neither 'patient' nor 'doctor'.

    Returns:
    None. Displays the plot and optionally saves it.
    """

    if who == "patient":
        i=0
    elif who == "doctor":
        i=1
    else:
        raise ValueError("'who' must be either 'patient' or 'doctor'")
    # Création de la heatmap avec des labels pour les cases
    plt.figure(figsize=(8, 6))  
    sns.heatmap(
        matrix[i],
        annot=True,            # Affiche les valeurs dans les cases
        cmap='plasma',        # Choix de la palette de couleurs
        cbar=True,             # Affiche la barre de couleurs
        yticklabels=xticklabel,    # Noms pour les colonnes
        xticklabels=yticklabel,     # Noms pour les lignes
        fmt=".2f",
    )
    
    plt.ylabel('dimension de génération')
    plt.xlabel("dimension d'estimation")

    if save == True:
        plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        
    # Ajout du titre de la heatmap
    plt.title(title)
    
    # Affichage de la heatmap
    plt.show()

def plot_tab_ami_inertia_kmeans(tab, save=False, title='Inertie et AMI en fonction du K du K-means, pour les patients et les docteurs'):
    """
    Plot inertia and AMI (Adjusted Mutual Information) scores for K-means clustering results.

    Parameters:
    - tab (pandas.DataFrame): DataFrame containing 'K', 'Inertia doctor', 'AMI doctor', 
                              'Inertia patient', and 'AMI patient' columns.
    - save (bool): If True, saves the plot as an image file. Default is False.
    - title (str): Main title for the plot. 
                   Default is 'Inertie et AMI en fonction du K du K-means, pour les patients et les docteurs'.

    Returns:
    None. Displays the plot and optionally saves it.
    """
    # Create the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot for doctor
    ax1 = axs[0]
    ax1.plot(tab['K'], tab['Inertia doctor'], 'r-o', label='Inertia')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Doctor')
    
    # Create twin axis for AMI Score for doctor
    ax2 = ax1.twinx()
    ax2.plot(tab['K'], tab['AMI doctor'], 'b-o', label='AMI Doctor')
    ax2.set_ylabel('AMI Score')
    
    # Add legends for both axes
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.85), ncol=1)
 
    # Plot for patient
    ax3 = axs[1]
    ax3.plot(tab['K'], tab['Inertia patient'], 'r-o', label='Inertia')
    ax3.set_xlabel('K')
    ax3.set_ylabel('Inertia')
    ax3.set_title('Patient')
    
    # Create twin axis for AMI Score for patient
    ax4 = ax3.twinx()
    ax4.plot(tab['K'], tab['AMI patient'], 'b-o', label='AMI Patient')
    ax4.set_ylabel('AMI Score')
    
    # Add legends for both axes
    ax3.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    ax4.legend(loc='upper right', bbox_to_anchor=(1, 0.85), ncol=1)

    # Adjust layout to make room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure if required
    if save:
        plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')

    # Add a main title for the figure
    plt.suptitle(title, fontsize=16)
    
    # Show the plot
    plt.show()

def plot_simulation_density_ami(matrix, rank=7, n=100, worst=1, save=True):
    """
    Plot simulation results for density and AMI (Adjusted Mutual Information).

    Parameters:
    - matrix (numpy.ndarray): 3D array containing simulation results.
    - rank (int): Number of different beta distances. Default is 7.
    - n (int): Number of simulations. Default is 100.
    - worst (int): Rank of the worst AMI to plot. Default is 1 (worst AMI).
    - save (bool): If True, saves the plot as an image file. Default is True.

    Returns:
    None. Displays the plot and optionally saves it.
    """
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
    ax4.set_ylabel("RMSE")
    ax4.set_xlabel('density')
    
    ax5.plot(mean_density_array, worst_ami_p, c="blue", label="patient")
    ax5.plot(mean_density_array, worst_ami_d, c="orange", label = "doctor")
    ax5.set_ylim(0,1)
    ax5.set_ylabel("AMI")
    ax5.set_xlabel('density')
    ax5.legend()
    
    if save:
        plt.savefig('density_ami.png', dpi=300, bbox_inches='tight')

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
    
@decorateur.compute_time
def plot_ef_RD(graph_object, ef_patient, ef_doctor, label_doctor = None, label_patient = None, RD_method = "PCA", save = True):
    
    """
    Visualise les embeddings fixes (EF) des patients et des docteurs après réduction de dimensionnalité.

    Parameters:
    - graph_object: Objet contenant les informations sur le graphe (doit avoir psi_class et alpha_class).
    - ef_patient (array): Embeddings fixes des patients.
    - ef_doctor (array): Embeddings fixes des docteurs.
    - label_doctor (array, optional): Labels additionnels pour les docteurs.
    - label_patient (array, optional): Labels additionnels pour les patients.
    - RD_method (str): Méthode de réduction de dimensionnalité ('PCA', 'TSNE', 'UMAP', ou 'Pacmap'). Par défaut 'PCA'.
    - save (bool): Si True, sauvegarde les graphiques. Par défaut True.

    Returns:
    None. Affiche et optionnellement sauvegarde les graphiques.
    """
    RD = {
        "PCA": PCA(n_components=2),
        "TSNE": TSNE(n_components=2),
        "UMAP": umap.UMAP(n_components=2),
    }

    if RD_method not in RD:
        raise ValueError(f"Méthode de réduction de dimensionnalité '{RD_method}' non reconnue.")

    ef_patient_RD = RD[RD_method].fit_transform(ef_patient)
    ef_doctor_RD = RD[RD_method].fit_transform(ef_doctor)
    
    #Visualisation des EF docteur
    sns.set_palette("deep") 
    
    # La couleur représente le degré, c'est-à-dire le nombre de lien qu'a le docteur
    sns.scatterplot( x= ef_doctor_RD[:,0], y=ef_doctor_RD[:,1], hue=graph_object.psi_class, palette='viridis', style = label_doctor)
    if save == True:
        plt.savefig("EF_doctor_" + RD_method)
    plt.title(f"representation 2D de l'embedding estimé des docteurs avec leur type en couleur (réduction par {RD_method})")
    plt.show()
    
    #Visualisation des EF patient
    sns.set_palette("deep") 
    
    # La couleur représente le degré, c'est-à-dire le nombre de lien qu'a le docteur
    sns.scatterplot( x= ef_patient_RD[:,0], y=ef_patient_RD[:,1], hue=graph_object.alpha_class, palette='viridis',  style = label_patient)

    if save == True:
        plt.savefig("EF_patient_" + RD_method)
    plt.title(f"representation 2D de l'embedding estimé des patients avec leur type en couleur (réduction par {RD_method})")
    plt.show()

def plot_ef_1D(graph_object, ef_patient, ef_doctor, save = False, title = "EF_doctor_patient_1D", style_d = None, style_p = None):
    """
    Visualise les embeddings fixes (EF) 1D des patients et des docteurs avec un jitter vertical.

    Parameters:
    - graph_object: Objet contenant les informations sur le graphe (doit avoir psi_class et alpha_class).
    - ef_patient (array): Embeddings fixes 1D des patients.
    - ef_doctor (array): Embeddings fixes 1D des docteurs.
    - save (bool): Si True, sauvegarde le graphique. Par défaut False.
    - title (str): Titre du graphique. Par défaut "EF_doctor_patient_1D".
    - style_d (array, optional): Style additionnel pour les points des docteurs.
    - style_p (array, optional): Style additionnel pour les points des patients.

    Returns:
    None. Affiche et optionnellement sauvegarde le graphique.
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  
    
    ef_patient, ef_doctor = ef_patient.flatten(), ef_doctor.flatten()
    
    # First plot: EF docteur
    sns.set_palette("deep")
    sns.scatterplot(ax=axes[0], x=ef_doctor, y=np.random.normal(0, 0.05, size=ef_doctor.shape[0]), 
                    hue=graph_object.psi_class, palette='viridis', style = style_d)
    #axes[0].set_title("Representation 1D de l'embedding estimé des docteurs avec leur type en couleur")
    axes[0].set_ylim(-1, 1)
    axes[0].set_xlabel("Valeur de l'effet fixe")
    axes[0].set_ylabel('Jitter')
    axes[0].legend(title="Classe")
    
    # Second plot: EF patients
    sns.scatterplot(ax=axes[1], x=ef_patient, y=np.random.normal(0, 0.1, size=ef_patient.shape[0]), 
                    hue=graph_object.alpha_class, palette='viridis', style = style_p )
    #axes[1].set_title("Representation 1D de l'embedding estimé des patients avec leur type en couleur (avec jitter)")
    axes[1].set_ylim(-1, 1)
    axes[1].set_xlabel("Valeur de l'effet fixe")
    axes[1].set_ylabel('Jitter')
    axes[1].legend(title="Classe")
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight') 
    plt.suptitle(title)
    # Show the combined figure
    plt.show()

def plot_ef_2D(graph_object, ef_patient, ef_doctor, save = False, ylim = [-1,1], title = "EF_doctor_patient_2D", style_d = None, style_p = None):
    """
    Visualise les embeddings fixes (EF) 2D des patients et des docteurs.

    Parameters:
    - graph_object: Objet contenant les informations sur le graphe (doit avoir psi_class et alpha_class).
    - ef_patient (array): Embeddings fixes 2D des patients.
    - ef_doctor (array): Embeddings fixes 2D des docteurs.
    - save (bool): Si True, sauvegarde le graphique. Par défaut False.
    - title (str): Titre du graphique. Par défaut "EF_doctor_patient_2D".
    - style_d (array, optional): Style additionnel pour les points des docteurs.
    - style_p (array, optional): Style additionnel pour les points des patients.

    Returns:
    None. Affiche et optionnellement sauvegarde le graphique.
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # First plot: EF docteur
    sns.set_palette("deep")
    sns.scatterplot(ax=axes[0], x=ef_doctor[:,0], y=ef_doctor[:,1], 
                    hue=graph_object.psi_class, palette='viridis', style = style_d)
    #axes[0].set_title("Representation 1D de l'embedding estimé des docteurs avec leur type en couleur")
    axes[0].set_ylim(ylim[0], ylim[1])
    axes[0].set_xlabel("axe x de l'effet fixe")
    axes[0].set_ylabel("axe y de l'effet fixe")
    axes[0].legend(title="Classe")
    
    # Second plot: EF patients
    sns.scatterplot(ax=axes[1], x=ef_patient[:,0], y=ef_patient[:,1], 
                    hue=graph_object.alpha_class, palette='viridis', style = style_p )
    #axes[1].set_title("Representation 1D de l'embedding estimé des patients avec leur type en couleur (avec jitter)")
    axes[1].set_ylim(ylim[0], ylim[1])
    axes[1].set_xlabel("axe x de l'effet fixe")
    axes[1].set_ylabel("axe y de l'effet fixe")
    axes[1].legend(title="Classe")
    
    # Adjust layout for better spacing
    plt.tight_layout() 
    if save:
        plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
    plt.suptitle(title)
    # Show the combined figure
    plt.show()

def plot_hist_degree(graph_object, save = True, hue_class = True):
    
    # Filtrer les lignes où y == 1
    df_links = graph_object.df[graph_object.df['y'] == 1]
    
    # Compter les degrés pour chaque agent dans le groupe 0 avec leur classe
    degree_group_p = df_links.groupby(['i', 'class_p']).size().reset_index(name='degree')
    degree_group_p.columns = ['agent', 'classe', 'degree']
    
    # Compter les degrés pour chaque agent dans le groupe 1 avec leur classe
    degree_group_d = df_links.groupby(['j', 'class_d']).size().reset_index(name='degree')
    degree_group_d.columns = ['agent', 'classe', 'degree']

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    if hue_class == False:
        sns.histplot(degree_group_p, x='degree', kde=False, ax=axs[0], bins = 20, binwidth= 1)
        axs[0].set_title('Histogramme des degrés pour les patients')
        axs[0].set_xlabel('Degré')
        axs[0].set_ylabel('Fréquence')
        
        # Tracer l'histogramme des docteurs
        sns.histplot(degree_group_d, x='degree', kde=False, ax=axs[1], bins = 15)
        axs[1].set_title('Histogramme des degrés pour les docteurs')
        axs[1].set_xlabel('Degré')
        axs[1].set_ylabel('Fréquence')

    else:
        # Tracer l'histogramme pour les patients avec distinction par classe
        sns.histplot(degree_group_p, x='degree', hue='classe', kde=True, ax=axs[0], palette = "viridis", bins = 50, multiple = "dodge", binwidth = 1)
        axs[0].set_title('Histogramme des degrés pour les patients avec la classe en couleur')
        axs[0].set_xlabel('Degré')
        axs[0].set_ylabel('Fréquence')
        
        # Tracer l'histogramme pour les docteurs avec distinction par classe
        sns.histplot(degree_group_d, x='degree', hue='classe', kde=True, ax=axs[1], palette = "viridis", bins=10, multiple = "dodge")
        axs[1].set_title('Histogramme des degrés pour les docteurs avec la classe en couleur')
        axs[1].set_xlabel('Degré')
        axs[1].set_ylabel('Fréquence')

    if save == True:
        plt.savefig("Histogrammes des degrés des patients et des docteurs")

    # Ajuster l'échelle de l'axe x pour être graduée par des entiers
    axs[0].set_xticks(range(int(axs[0].get_xlim()[0]), int(axs[0].get_xlim()[1]) + 1))  # Définir des ticks à chaque entier
    
    # Afficher les graphiques
    plt.tight_layout()
    plt.show()


@decorateur.compute_time
def plot_rmse_epochs(sim_beta_distance_array = [-25,-20, -17, -15,-12,-10, -8, -7,], epochs_step = 10, n=10, save=True, algorithm = "MF"):

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
                estimates = get_estimations( graph_object.df, nb_epochs=epochs_step, show_print=0,  initial_weights=None, algorithm = algorithm )
            else:
                estimates = get_estimations( graph_object.df, nb_epochs=epochs_step, show_print=0,  initial_weights=estimates[1], algorithm = algorithm )
            alpha_hat = estimates[1][0]
            psi_hat = estimates[1][1]
            rmse_array[i] = metrique.rmse(alpha_hat = alpha_hat, psi_hat = psi_hat, alpha_star = graph_object.alpha_graph, psi_star = graph_object.psi_graph)[0]
        plt.plot(np.arange(epochs_step, epochs_step*(n+1), epochs_step), rmse_array, label=f"beta distance = {sim_beta_distance}/ density = {graph_object.density*100:.2f}%")

    plt.title("évolution du RMSE des effets fixes à travers \n les epoques pour différentes densités")
    plt.xlabel("epochs")
    plt.ylabel('RMSE'),
    plt.legend()
    if save==True:
        # Sauvegarder le graphique en tant qu'image PNG
        plt.savefig('RMSE_vs_epochs_density.png')
   
    plt.show()      


@decorateur.compute_time
def plot_rmse_epochs_multiD(sim_beta_distance_array = [-25,-20, -17, -15,-12,-10, -8, -7,], epochs_step = 10, n=10, save=True, alpha_law_means=[[0,1], [1,-1], [2,0]], psi_law_means= [[1,1], [1,-1], [-1,2]], gaussian_sphere=False, std_multiplier_p=0, std_multiplier_d=0, nb_latent_factors=2, radius=4, dilatation_p=1.5, dilatation_d = 1.5, seed = 12, algorithm = "MF"):

    identity_matrices = np.array([np.eye(nb_latent_factors) for _ in range(3)])
    
    for sim_beta_distance in sim_beta_distance_array:
        
        rmse_array = np.zeros(n)
        graph_object= GraphFormation(
                        n_patients=1000,
                         n_doctors=50,
                        alpha_law_means= alpha_law_means,
                        psi_law_means = psi_law_means,
                        std_multiplier_d= std_multiplier_d,
                        std_multiplier_p = std_multiplier_p,
                        gaussian_sphere = gaussian_sphere,
                        nb_latent_factors=nb_latent_factors,
                        beta_distance_graph = sim_beta_distance,
                        dilatation_p=dilatation_p,
                        dilatation_d=dilatation_p,
                        radius=radius,
                        psi_law_stds= identity_matrices,
                        alpha_law_stds = identity_matrices,
                        seed = seed
                        )
        graph_object.do_the_graph()
        
        for i,step in enumerate(np.arange(epochs_step, epochs_step*(n+1), epochs_step)):
            if i==0:
                estimates = get_estimations( graph_object.df, nb_epochs=epochs_step, show_print=0,  initial_weights=None, dim_embedding = nb_latent_factors, algorithm = algorithm )
            else:
                estimates = get_estimations( graph_object.df, nb_epochs=epochs_step, show_print=0,  initial_weights=estimates[1], dim_embedding = nb_latent_factors, algorithm  = algorithm )
            alpha_hat = estimates[1][0]
            psi_hat = estimates[1][1]
            rmse_array[i] = metrique.rmse_multiD(alpha_hat = alpha_hat, psi_hat = psi_hat, alpha_star = graph_object.alpha_graph, psi_star = graph_object.psi_graph)
        plt.plot(np.arange(epochs_step, epochs_step*(n+1), epochs_step), rmse_array, label=f"beta distance = {sim_beta_distance}/ density = {graph_object.density*100:.2f}%")

    plt.title("évolution du RMSE des effets fixes à travers \n les epoques pour différentes densités")
    plt.xlabel("epochs")
    plt.ylabel('RMSE multidimensionnel')
    plt.legend()
    if save==True:
        # Sauvegarder le graphique en tant qu'image PNG
        plt.savefig('RMSE_vs_epochs_density_multiD.png')
   
    plt.show()  


#### !!! Ancienne fontion à modifier !!! #######
@decorateur.compute_time
def plot_rmsedensity( sim_beta_distance_array = [-25, -20,-15, -12, -10,-8,-5, -3,], nb_epochs = 200):

    """
    Cette fonction trace l'évolution du RMSE (Root Mean Square Error) des effets fixes à travers les époques 
    pour différentes densités de graphes simulés selon des distances bêta.

    Paramètres:
    ----------
    sim_beta_distance_array : list of float, default=[-25, -20, -17, -15, -12, -10, -8, -7]
        Liste des valeurs de distance bêta utilisées pour la simulation des graphes.
    
    epochs_step : int, default=10
        Nombre d'époques entre chaque étape de calcul du RMSE.
    
    n : int, default=10
        Nombre total de points (étapes) à calculer pour chaque simulation de graphe.
    
    save : bool, default=True
        Indicateur pour sauvegarder ou non le graphique généré sous forme de fichier PNG.
    
    alpha_law_means : list of list of int, default=[[0,1], [1,-1], [2,0]]
        Moyennes des lois normales pour les alphas des patients.

    psi_law_means : list of list of int, default=[[1,1], [1,-1], [-1,2]]
        Moyennes des lois normales pour les psis des médecins.
    
    gaussian_sphere : bool, default=False
        Indicateur pour déterminer si les facteurs latents sont générés sur une sphère gaussienne.
    
    std_multiplier_p : float, default=0
        Multiplicateur de l'écart-type pour la distribution des patients.
    
    std_multiplier_d : float, default=0
        Multiplicateur de l'écart-type pour la distribution des médecins.
    
    nb_latent_factors : int, default=2
        Nombre de facteurs latents dans le modèle.

    radius : float, default=4
        Rayon utilisé pour la construction du graphe.
    
    dilatation_p : float, default=1.5
        Facteur de dilatation pour les patients.

    dilatation_d : float, default=1.5
        Facteur de dilatation pour les médecins.
    
    seed : int, default=12
        Graine aléatoire pour la reproductibilité des simulations.

    algorithm : str, default="MF"
        Algorithme utilisé pour l'estimation des paramètres du modèle.

    Retour:
    -------
    None
        La fonction ne retourne rien mais affiche et (optionnellement) sauvegarde un graphique.
    """
    
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

        rmse_array[i] = metrique.rmse(ef_patient_hat, ef_doctor_hat, alpha_star, psi_star)[0]
        rmse_alpha_array[i] = metrique.rmse_alpha(ef_patient_hat, alpha_star)[0]
        rmse_psi_array[i] = metrique.rmse_psi(ef_doctor_hat, psi_star)[0]

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
