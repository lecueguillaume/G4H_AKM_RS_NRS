######## SCRIPT POUR LES SIMULATIONS ###########

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
import script.metrique as metrique

GraphFormation = graph_formation.GraphFormation
get_estimations = MF_NRS.get_estimations

# Clustering
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity


@decorateur.log_execution_time('execution_details.txt')
def make_matrix_heatmap(dim_gen_array = np.array([1,2,3,4,5]), dim_estim_array = np.array([1, 2, 3, 4, 5, 8, 10]), seed = 12, algorithm = "NRS", nb_epochs_max = 100, name_file = "matrix_heatmap"):

    matrix = np.zeros((2, len(dim_gen_array), len(dim_estim_array)))

    for i,dim_gen in enumerate(dim_gen_array):
        if dim_gen == 1:
            
            graph_object= GraphFormation(n_patients=1000, n_doctors=50, max_number_connections=50, beta_distance_graph = -25, seed = seed)
            graph_object.do_the_graph()
            
        elif dim_gen == 2:
                
            alpha_law_means = [[0,1], [1,-1], [2,0]]
            psi_law_means = [[1,1], [1,-1], [-1,2]]
            identity_matrices = np.array([np.eye(2) for _ in range(max(len(graph_object.alpha_law_weights), len(graph_object.psi_law_weights)) )])  
            graph_object= GraphFormation(
                                n_patients=1000,
                                 n_doctors=50,
                                 max_number_connections=50,
                                beta_distance_graph = -20,
                                dilatation_p = 1.5,
                                dilatation_d = 1.5,
                                alpha_law_means = alpha_law_means,
                                psi_law_means = psi_law_means,
                                psi_law_stds= identity_matrices,
                                alpha_law_stds = identity_matrices,
                                std_multiplier_p=0,
                                std_multiplier_d=0,
                                nb_latent_factors=2,
                                seed = seed)
            graph_object.do_the_graph()

        else:
            
            identity_matrices = np.array([np.eye(dim_gen) for _ in range(max(len(graph_object.alpha_law_weights), len(graph_object.psi_law_weights)) )])
            graph_object= GraphFormation(
                                n_patients=1000,
                                 n_doctors=50,
                                 max_number_connections=50,
                                beta_distance_graph = -25,
                                psi_law_stds= identity_matrices,
                                alpha_law_stds = identity_matrices,
                                std_multiplier_p=0.3,
                                std_multiplier_d=0.5,
                                nb_latent_factors= dim_gen,
                                gaussian_sphere = True,
                                radius = 4,
                                seed=seed)
            graph_object.do_the_graph()

        print("graph formation: done")
        for j, dim_estim in enumerate(dim_estim_array):

            # First estimation to determine the optimal number of epochs
            estimates = get_estimations(graph_object.df, nb_epochs=nb_epochs_max, show_print=0, seed=seed, dim_embedding=dim_estim, valid_split=0.15,  algorithm =  algorithm)
            history = estimates[5]
            min_val_loss = min(history.history["val_loss"])
            min_val_loss_epoch = history.history["val_loss"].index(min_val_loss)

            print("first estimation: done")
            # Second estimation with the right number of epochs
            estimates = get_estimations(graph_object.df, nb_epochs=min_val_loss_epoch, show_print=0, seed=seed, dim_embedding=dim_estim, valid_split=0.0,  algorithm =  algorithm)
            print("second estimation: done")

            ef_patient = estimates[1][0]
            ef_doctor = estimates[1][1]

            # Check
            print(ef_patient.shape)
            print(ef_doctor.shape)
        
            if ef_patient.shape[1] == 1:
                ef_patient = ef_patient.flatten().reshape(-1,1)
            if ef_doctor.shape[1] == 1:
                ef_doctor = ef_doctor.flatten().reshape(-1,1)

            # K-means sur les patients avec le bon nombre K
            kmeans_p = KMeans(n_clusters= len(graph_object.alpha_law_weights), random_state=seed)
            labels_p = kmeans_p.fit_predict(ef_patient)

            # K-means sur les docteurs avec le bon nombre K
            kmeans_d = KMeans(n_clusters= len(graph_object.psi_law_weights), random_state=seed)
            labels_d = kmeans_p.fit_predict(ef_doctor)

            print("clustering: done")

            matrix[0][i][j] = metrics.adjusted_rand_score(graph_object.alpha_class, labels_p)
            matrix[1][i][j] = metrics.adjusted_rand_score(graph_object.psi_class, labels_d)

    with open( name_file +'.npy', 'wb') as f:
        np.save(f, matrix)


    return matrix

                


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
    
            rmse_array[i] = metrique.rmse(ef_patient_hat, ef_doctor_hat, graph_object.alpha_graph, graph_object.psi_graph)[0]
            #rmse_alpha_array[i] = metrique.rmse_alpha(ef_patient_hat, alpha_star)[0]
            #rmse_psi_array[i] = metrique.rmse_psi(ef_doctor_hat, psi_star)[0]
    
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



@decorateur.log_execution_time('execution_details.txt')
def simulation_rmse_lambda( regu_lambdas = np.array([10**(-i) for i in range(5,11)]), nb_epochs=200, save=True ):
    
    size = len(regu_lambdas)
    rmse_array, rmse_alpha_array, rmse_psi_array = np.zeros(size), np.zeros(size), np.zeros(size)
    
    for i,regu_lambda in enumerate(regu_lambdas):
        graph_object= GraphFormation(
                        n_patients=1000,
                         n_doctors=50,
                         max_number_connections=50,
                        beta_distance_graph = -25,)
        graph_object.do_the_graph()

        estimates =  get_estimations(graph_object.df, nb_epochs=nb_epochs, seed=12, l_lambda = regu_lambda, )
        
        alpha_hat = np.array(estimates[1][0])
        psi_hat = np.array(estimates[1][1])

        rmse_array[i] =  metrique.rmse(alpha_hat, psi_hat, alpha_star = graph_object.alpha_graph, psi_star = graph_object.psi_graph)[0]
        rmse_alpha_array[i] =  metrique.rmse_alpha(alpha_hat, graph_object.alpha_graph)[0]
        rmse_psi_array[i] =  metrique.rmse_psi(psi_hat, graph_object.psi_graph)[0]

        print(f"Simulation {i+1}: done !")

    x = -np.log10(regu_lambdas)
    # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))
    
    axs[0].set_title(f"Rmse en fonction du lambda (10^(-i) pour l'échelle) de régularization l2" )
    axs[0].plot(x, rmse_array)
    axs[0].set_xlabel('lambda (10^(-x))')
    axs[0].set_ylabel('RMSE')

    axs[0].set_title(f"Rmse des alpha et des psi en fonction du lambda (10^(-i) pour l'échelle) de régularization l2" )
    axs[1].plot(x, rmse_alpha_array, label="RMSE of psi")
    axs[1].plot(x, rmse_psi_array, label="RMSE of alpha")
    axs[1].set_xlabel('lambda (10^(-x))')
    axs[1].set_ylabel('RMSE')
    axs[1].legend()
    
    if save == True:
        plt.savefig('Simulation_regularization_lambda.png')
    plt.tight_layout()
    plt.show()    

@decorateur.log_execution_time('execution_details.txt')
def simulation_ami_lambda( regu_lambdas = np.array([10**(-i) for i in range(5,11)]), nb_epochs=150, save=True, algorithm = "MF" ):
    
    size = len(regu_lambdas)
    ami_alpha_array, ami_psi_array = np.zeros(size), np.zeros(size)
    
    for i,regu_lambda in enumerate(regu_lambdas):
        graph_object= GraphFormation(
                        n_patients=1000,
                         n_doctors=50,
                         max_number_connections=50,
                        beta_distance_graph = -25,)
        graph_object.do_the_graph()

        estimates =  get_estimations(graph_object.df, nb_epochs=nb_epochs, seed=12, l_lambda = regu_lambda, algorithm = algorithm)

        ef_patient = estimates[1][0]
        ef_doctor = estimates[1][1]
        
        if ef_patient.shape[1] == 1:
            ef_patient = ef_patient.flatten().reshape(-1,1)
        if ef_doctor.shape[1] == 1:
            ef_doctor = ef_doctor.flatten().reshape(-1,1)

        # KMeans for Group patients
        kmeans_p = KMeans(n_clusters=3, random_state=0)
        labels_p = kmeans_p.fit_predict(ef_patient)
        ami_alpha_array[i] = metrics.adjusted_rand_score(graph_object.alpha_class, labels_p)
        
        # KMeans for doctors
        kmeans_d = KMeans(n_clusters=3, random_state=0)
        labels_d = kmeans_d.fit_predict(ef_doctor)
        ami_psi_array[i] = metrics.adjusted_rand_score(graph_object.psi_class, labels_d)
        
        print(f"Simulation {i+1}: done !")

    x = -np.log10(regu_lambdas)
    # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(1, 1, figsize=(10, 15))

    axs.set_title(f"AMI des patients et des docteurs en fonction du lambda (10^(-i) pour l'échelle) de régularization l2")
    axs.plot(x, ami_psi_array, label="AMI docteur")
    axs.plot(x, ami_alpha_array, label="AMI patient")
    axs.set_xlabel('lambda (10^(-x))')
    axs.set_ylabel('AMI')
    axs.legend()
    
    if save == True:
        plt.savefig('Simulation_regularization_lambda_ami.png')
    plt.tight_layout()
    plt.show()  

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

@decorateur.compute_time
def robustesse_simul_n_models(n, train_ratio, graph_object, nb_epochs = 80, dim_embedding = 1,  algorithm = "MF"):

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
   

    Exemple :
    --------
    graph_obj = GraphObject(...)
    matrix_ef_patient, matrix_ef_doctor, beta_X_p, beta_X_d, beta_D_p, beta_D_d, beta_distance = robustesse_simul_n_models(100, 0.8, graph_obj)
    """
    
    # On va découper le df en 2 à partir de l'indice "train_indices"
    df = graph_object.df
    train_indices = int(df.shape[0]*train_ratio)
    
    matrix_ef_patient, matrix_ef_doctor = np.zeros((n, graph_object.n_patients, dim_embedding)), np.zeros((n,graph_object.n_doctors, dim_embedding))
    
    for k in range(n):
        df_shuffled = df.sample(frac=1, random_state=k+1).reset_index(drop=True)
        df_train, df_test = df_shuffled.iloc[:train_indices,:] , df_shuffled.iloc[train_indices:,:]
        
        estimates = get_estimations(df_train, seed = k+1, dim_embedding = dim_embedding, nb_epochs= nb_epochs, algorithm = algorithm)
        
        matrix_ef_patient[k] = estimates[1][0]
        matrix_ef_doctor[k] = estimates[1][1]
       
        print(f"simulation {k+1}: done")

    return matrix_ef_patient, matrix_ef_doctor      


def make_distance_ef_matrix(ef):
    size = ef.shape[0]
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i):
            if ef.shape[1] == 1:
                matrix[i][j] = np.abs(ef[i] - ef[j])
            else:
                matrix[i][j] = np.linalg.norm(ef[i] - ef[j])
    return matrix

def make_list_dist_ef_matrix(matrix_ef_simulation, triangle=False):
    l = []
    for n in range(matrix_ef_simulation.shape[0]):
        ef = matrix_ef_simulation[n]
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

def robustesse_all(n, train_ratio, graph_object, nb_epochs = 80, dim_embedding = 1,  algorithm = "MF"):
    matrix_ef_patient, matrix_ef_doctor = robustesse_simul_n_models(n, train_ratio, graph_object, nb_epochs = nb_epochs, dim_embedding = dim_embedding,  algorithm = algorithm)
    list_dist_matrix_patient = make_list_dist_ef_matrix(matrix_ef_patient, triangle=False)
    list_dist_matrix_doctor = make_list_dist_ef_matrix(matrix_ef_doctor, triangle=False)

    print(f"Moyenne de la similarité des matrices de distances des ef patients : {compute_cosine_similarity(list_dist_matrix_patient, triangle=True).mean()}")
    print(f"Moyenne de la similarité des matrices de distances des ef doctors : {compute_cosine_similarity(list_dist_matrix_doctor, triangle=True).mean()}")

    # Calcul des moyennes de similarité
    moyenne_patient = compute_cosine_similarity(list_dist_matrix_patient, triangle=True).mean()
    moyenne_doctor = compute_cosine_similarity(list_dist_matrix_doctor, triangle=True).mean()
    
    # Contenu à écrire dans le fichier
    content = 100*'-' + "\n"
    content += f"n = {n}; train_ratio = {train_ratio}; nb_epochs = {nb_epochs}; dim_embedding = {dim_embedding}; algo = {algorithm}\n "
    content += f"Moyenne de la similarité des matrices de distances des ef patients : {moyenne_patient}\n"
    content += f"Moyenne de la similarité des matrices de distances des ef doctors : {moyenne_doctor}\n"
    content += 100*'-' + "\n"

    # Écriture dans un fichier texte
    with open('robustesse_results.txt', 'a') as file:
        file.write(content)




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

