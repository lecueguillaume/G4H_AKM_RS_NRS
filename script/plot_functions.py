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
import pacmap
import seaborn as sns
# Clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics

@decorateur.compute_time
def plot_ef_RD(graph_object, ef_patient, ef_doctor, label_doctor = None, label_patient = None, RD_method = "PCA", save = True):
    
    RD = {"PCA": PCA(n_components=2) , "TSNE" :  TSNE(n_components=2), "UMAP":umap.UMAP(n_components=2) , "Pacmap" :  pacmap.PaCMAP(n_dims=2,n_neighbors=7)}
    ef_patient_RD, ef_doctor_RD = RD[RD_method].fit_transform(ef_patient), RD[RD_method].fit_transform(ef_doctor)


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