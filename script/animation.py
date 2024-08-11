# bibliothéques fondamentales
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random
import matplotlib.animation as animation

# module maison
import script.decorateur as decorateur
import script.graph_formation as graph_formation
import script.MF_NRS as MF_NRS

from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

GraphFormation = graph_formation.GraphFormation
get_estimations = MF_NRS.get_estimations

#################################################################################################################################
####################################  Animation 1D  #############################################################################
#################################################################################################################################

@decorateur.compute_time
def animation_embedding_1D(graph_object, nb_epochs=100, who = 'patient' , interval=200):
    # Create figure and axis
    fig, ax = plt.subplots(1,1,figsize=(12, 12))

    
    # Initialize storage arrays
    A = np.zeros((nb_epochs, graph_object.n_patients))  
    B = np.zeros((nb_epochs, graph_object.n_doctors))  

    # Initialization
    estimates = get_estimations(graph_object.df, initial_weights=None, nb_epochs=1, valid_split = 0.01)
    
    ef_patient, ef_doctor = estimates[1][0], estimates[1][1]
    A[0], B[0] = ef_patient.ravel(), ef_doctor.ravel()
    jitter_p = np.random.normal(0, 0.05, size=ef_patient.shape[0])
    jitter_d = np.random.normal(0, 0.05, size=ef_doctor.shape[0])
    
    for i in range(1, nb_epochs):
        estimates = get_estimations(graph_object.df, initial_weights=estimates[1], nb_epochs=1, valid_split=0.01)
        ef_patient, ef_doctor = estimates[1][0], estimates[1][1]
        A[i], B[i] = ef_patient.ravel(), ef_doctor.ravel()
        
    if who == "patient":
        matrix = A
        jitter = jitter_p
        title = "Evolution des EF des patients"
        hue_values = np.array(graph_object.alpha_class)
    else:
        matrix = B
        jitter = jitter_d
        title = "Evolution des EF des doctors"
        hue_values = np.array(graph_object.psi_class)

    # Initial scatter plot (empty, to be updated in animation)
    scat = ax.scatter([], [], c=[], cmap='viridis', norm=plt.Normalize(hue_values.min(), hue_values.max()))
        
    # Create initial scatter plot
    scat = ax.scatter(matrix[0], jitter)
    ax.set(xlim=[-7,7], ylim=[-0.75, 0.75], xlabel='EF value', ylabel='jitter')
    ax.set_title(title)

    def update(frame):   
        # Update scatter plot with new data
        scat.set_offsets(np.c_[matrix[frame], jitter])
        scat.set_array(hue_values.astype(float))  # Update the color mapping  
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=nb_epochs, interval=interval, blit=True)

    # Save the animation as a GIF
    ani.save('animation_embedding_' + who +'.gif', writer='pillow')
    print("Save done")


#################################################################################################################################
####################################  Animation 2D  #############################################################################
#################################################################################################################################
        
@decorateur.compute_time       
def animation_embedding_2D(graph_object, nb_epochs=100, who = 'patient' , interval=200):
    
    # Create figure and axis
    fig, ax = plt.subplots(1,1,figsize=(12, 12))

    # Initialize storage arrays
    A = np.zeros((nb_epochs, graph_object.n_patients, 2))  
    B = np.zeros((nb_epochs, graph_object.n_doctors, 2))  

    # Initialization
    estimates = get_estimations(graph_object.df, initial_weights=None, nb_epochs=1,  dim_embedding=2)
    
    ef_patient, ef_doctor = estimates[1][0], estimates[1][1]
    A[0], B[0] = ef_patient, ef_doctor
    
    for i in range(1, nb_epochs):
        estimates = get_estimations(graph_object.df, initial_weights=estimates[1], nb_epochs=1, dim_embedding=2)
        ef_patient, ef_doctor = estimates[1][0], estimates[1][1]
        A[i], B[i] = ef_patient, ef_doctor
        
    if who == "patient":
        matrix = A
        title = "Evolution des EF des patients"
        hue_values = np.array(graph_object.alpha_class)
    else:
        matrix = B
        title = "Evolution des EF des doctors"
        hue_values = np.array(graph_object.psi_class)

    # Initial scatter plot (empty, to be updated in animation)
    scat = ax.scatter([], [], c=[], cmap='viridis', norm=plt.Normalize(hue_values.min(), hue_values.max()))
        
    # Create initial scatter plot
    scat = ax.scatter(matrix[0][:,0], matrix[0][:,1] )
    ax.set(xlim=[-7,7], ylim=[-7, 7], xlabel=' x axis EF value', ylabel='y axis EF value')
    ax.set_title(title)

    def update(frame):   
        # Update scatter plot with new data
        scat.set_offsets(np.c_[matrix[frame][:,0], matrix[frame][:,1]])
        scat.set_array(hue_values.astype(float))  # Update the color mapping  
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=nb_epochs, interval=interval, blit=True)

    # Save the animation as a GIF
    ani.save('animation_embedding_2D_' + who +'.gif', writer='pillow')
    print("Save done")


#################################################################################################################################
####################################  Animation XD avec réduction de dimension  #################################################
#################################################################################################################################
@decorateur.compute_time
def animation_embedding_XD(graph_object, RD_method = "PCA",  nb_epochs=100, who = 'patient' , interval=200, dimension=5):


    if RD_method == "PCA":
        RD_function = PCA(n_components=2)
    elif RD_method == "T-SNE":
        RD_function = TSNE(n_components=2)
    elif RD_method == "Isomap":
        RD_function = Isomap(n_components=2)
    else:
        print("Mettre PCA, T-SNE ou Isomap comme méthode de réduction de  dimension") 
        return None
        
    # Create figure and axis
    fig, ax = plt.subplots(1,1,figsize=(12, 12))

    # Initialize storage arrays
    A = np.zeros((nb_epochs, graph_object.n_patients, 2))  
    B = np.zeros((nb_epochs, graph_object.n_doctors, 2))  

    # Initialization
    estimates = get_estimations(graph_object.df, initial_weights=None, nb_epochs=1,  dim_embedding=dimension)
    ef_patient, ef_doctor = estimates[1][0], estimates[1][1]
    ef_patient_2D, ef_doctor_2D = RD_function.fit_transform(ef_patient), RD_function.fit_transform(ef_doctor)

    
    A[0], B[0] = ef_patient_2D, ef_doctor_2D
    
    for i in range(1, nb_epochs):
        estimates = get_estimations(graph_object.df, initial_weights=estimates[1], nb_epochs=1, dim_embedding=dimension)
        ef_patient, ef_doctor = estimates[1][0], estimates[1][1]
        ef_patient_2D, ef_doctor_2D = RD_function.fit_transform(ef_patient), RD_function.fit_transform(ef_doctor)
        A[i], B[i] = ef_patient_2D, ef_doctor_2D
        
    if who == "patient":
        matrix = A
        title = "Evolution des EF des patients (dimension reduced by " + RD_method + ")"
        hue_values = np.array(graph_object.alpha_class)
    else:
        matrix = B
        title = "Evolution des EF des doctors (dimension reduced by " + RD_method + ")"
        hue_values = np.array(graph_object.psi_class)

    # Initial scatter plot (empty, to be updated in animation)
    scat = ax.scatter([], [], c=[], cmap='viridis', norm=plt.Normalize(hue_values.min(), hue_values.max()))
        
    # Create initial scatter plot
    scat = ax.scatter(matrix[0][:,0], matrix[0][:,1] )
    ax.set(xlim=[-7,7], ylim=[-7, 7], xlabel=' x axis EF value', ylabel='y axis EF value')
    ax.set_title(title)

    def update(frame):   
        # Update scatter plot with new data
        scat.set_offsets(np.c_[matrix[frame][:,0], matrix[frame][:,1]])
        scat.set_array(hue_values.astype(float))  # Update the color mapping  
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=nb_epochs, interval=interval, blit=True)

    # Save the animation as a GIF
    ani.save(f'animation_embedding_{dimension}D_' + RD_method + '_' + who +'.gif', writer='pillow')
    print("Save done")
