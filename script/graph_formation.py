### SCRIPT pour la formation du graphe ###

# bibliothéques fondamentales
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random
pd.options.mode.chained_assignment = None  # default='warn' # Remove copy on slice warning

# Pour faire du ML/DL
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import preprocessing

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
simplefilter("ignore", category=ConvergenceWarning) # Useful for logistic regression

from tensorflow.keras.layers import Input, Dot, Embedding, Add, Flatten, Activation, Layer # Optimisation via tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
pd.options.mode.chained_assignment = None  # default='warn' # Remove copy on slice warning
import tensorflow as tf
from keras import regularizers
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Confusion Matrix

import numpy as np
import pandas as pd
import random
import tensorflow as tf

class GraphFormation:

    """
    A class used to represent the formation of a graph between patients and doctors.

    Attributes
    ----------
    n_patients : int
        The number of patients in the graph
    n_doctors : int
        The number of doctors in the graph
    max_number_connections : int
        The maximum number of connections a patient can have with doctors
    beta_X_p_graph : float
        The weight of the patient continious variable in the graph formation
    beta_X_d_graph : float
        The weight of the doctor continious variable in the graph formation
    beta_D_p_graph : float
        The weight of the patient binary variable in the graph formation
    beta_D_d_graph : float
        The weight of the doctor binary variable in the graph formation
    beta_distance_graph : float
        The weight of distance in the graph formation
    alpha_law_weights : np.ndarray
        The weights of the alpha law used in the graph
    alpha_law_means : np.ndarray
        The means of the alpha law used in the graph
    alpha_law_stds : np.ndarray
        The standard deviations of the alpha law used in the graph
    psi_law_weights : np.ndarray
        The weights of the psi law used in the graph
    psi_law_means : np.ndarray
        The means of the psi law used in the graph
    psi_law_stds : np.ndarray
        The standard deviations of the psi law used in the graph
    seed : int
        The seed used for random number generation
    nb_latent_factors : int
        The number of latent factors used in the graph formation
    no_seed_for_link : bool
        Whether to use a seed for link generation or not
    dilatation_p : int
        The dilatation factor for patients
    dilatation_d : int
        The dilatation factor for doctors
    std_multiplier_p : int
        The standard deviation multiplier for patients
    std_multiplier_d : int
        The standard deviation multiplier for doctors
    dataframe : pd.DataFrame
        The resulting dataframe after graph formation
    alpha_graph : list
        The alpha graph after graph formation
    psi_graph : list
        The psi graph after graph formation
    alpha_class : list
        The alpha class after graph formation
    psi_class : list
        The psi class after graph formation
    coor_patients : list
        The coordinates of patients
    coor_doctors : list
        The coordinates of doctors
    D : np.ndarray
        The distance matrix between patients and doctors
    sim_patient_X_normed : np.ndarray
        The normalized variable X of patients
    sim_doctor_X_normed : np.ndarray
        The normalized variable X of doctors
    sim_patient_D : np.ndarray
        The binary variable simulated for patients
    sim_doctor_D : np.ndarray
        The binary variable simulated for doctors

    Methods
    -------
    do_the_graph()
        Generates the graph based on the given parameters and attributes.
    """
        
    def __init__(self,
                 n_patients,
                 n_doctors,
                 max_number_connections=1000,
                 beta_X_p_graph=0.2,
                 beta_X_d_graph=0.2,
                 beta_D_p_graph=1,
                 beta_D_d_graph=1,
                 beta_distance_graph=-20,
                 alpha_law_weights=[0.3, 0.3, 0.4],
                 alpha_law_means=np.array([0, 1, 2]),
                 alpha_law_stds=np.array([1, 1, 1]),
                 psi_law_weights=[0.3, 0.4, 0.3],
                 psi_law_means=np.array([0, 1, 2]),
                 psi_law_stds=np.array([1, 1, 1]),
                 seed=12,
                 nb_latent_factors=1,
                 no_seed_for_link=True,
                 dilatation_p=2,
                 dilatation_d=2,
                 std_multiplier_p=0,
                 std_multiplier_d=0,
                 gaussian_sphere = False,
                 radius=1):

        self.n_patients = n_patients
        self.n_doctors = n_doctors
        self.max_number_connections = max_number_connections
        self.beta_X_p_graph = beta_X_p_graph
        self.beta_X_d_graph = beta_X_d_graph
        self.beta_D_p_graph = beta_D_p_graph
        self.beta_D_d_graph = beta_D_d_graph
        self.beta_distance_graph = beta_distance_graph
        self.alpha_law_weights = np.array(alpha_law_weights)
        self.alpha_law_means = np.array(alpha_law_means)
        self.alpha_law_stds = np.array(alpha_law_stds)
        self.psi_law_weights = np.array(psi_law_weights)
        self.psi_law_means = np.array(psi_law_means)
        self.psi_law_stds = np.array(psi_law_stds)
        self.seed = seed
        self.nb_latent_factors = nb_latent_factors
        self.no_seed_for_link = no_seed_for_link
        self.dilatation_p = dilatation_p
        self.dilatation_d = dilatation_d
        self.std_multiplier_p = std_multiplier_p
        self.std_multiplier_d = std_multiplier_d
        self.gaussian_sphere = gaussian_sphere
        self.radius = radius
        
        # Initialize attributes
        self.df = None
        self.alpha_graph = None
        self.psi_graph = None
        self.alpha_class = None
        self.psi_class = None
        self.coor_patients = None
        self.coor_doctors = None
        self.D = None
        self.sim_patient_X_normed = None
        self.sim_doctor_X_normed = None
        self.sim_patient_D = None
        self.sim_doctor_D = None 
        self.density = None
        self.P = None

    def do_the_graph(self, show_time_execution=False): 
        
        """
        parameter:
        show_time_execution : bool
        Whether to show the time of execution of the graph formation or not
        
        Generates the graph based on the given parameters and attributes.
        This method initializes the random seeds, generates fixed effects, coordinates, and the distance matrix.
        It also generates continious variable and other data for patients and doctors, compiles IDs and features, and generates
        the connection matrix. Finally, it compiles relations into a dataframe and adjusts the number of connections.

        Returns
        -------
        None
        """
    
        start_time = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
        coor_patients = []
        coor_doctors = []
        alpha_graph = np.zeros((self.n_patients, self.nb_latent_factors))
        alpha_class = []
        psi_graph = np.zeros((self.n_doctors, self.nb_latent_factors))
        psi_class = []
        rng = np.random.default_rng(self.seed)
        D = np.zeros([self.n_patients, self.n_doctors])

        # Adjust means and stds
        alpha_law_means = self.alpha_law_means * self.dilatation_p
        alpha_law_stds = self.alpha_law_stds * self.std_multiplier_p
        psi_law_means = self.psi_law_means * self.dilatation_d
        psi_law_stds = self.psi_law_stds * self.std_multiplier_d

        # Normalize weights
        self.alpha_law_weights /= self.alpha_law_weights.sum()
        self.psi_law_weights /= self.psi_law_weights.sum()

        nb_class_alpha = len(self.alpha_law_weights)
        nb_class_psi = len(self.psi_law_weights)

        if self.nb_latent_factors == 1:
            # Generate fixed effects
            for i in range(self.n_patients):
                chosen_class = np.random.choice(np.arange(nb_class_alpha), p=self.alpha_law_weights)
                alpha_class.append(chosen_class)
                fe_vector = np.random.normal(alpha_law_means[chosen_class], alpha_law_stds[chosen_class], size=1)
                alpha_graph[i] =  fe_vector
    
            for j in range(self.n_doctors):
                chosen_class = np.random.choice(np.arange(nb_class_psi), p=self.psi_law_weights)
                psi_class.append(chosen_class)
                fe_vector = np.random.normal(psi_law_means[chosen_class], psi_law_stds[chosen_class], size=1)
                psi_graph[j] = fe_vector

        elif self.gaussian_sphere == False:        
            # Generate fixed effects
            for i in range(self.n_patients):
                chosen_class = np.random.choice(np.arange(nb_class_alpha), p=self.alpha_law_weights)
                alpha_class.append(chosen_class)
                ef_vector = np.random.multivariate_normal(alpha_law_means[chosen_class], alpha_law_stds[chosen_class], size=1)
                alpha_graph[i, :] = ef_vector
    
            for j in range(self.n_doctors):
                chosen_class = np.random.choice(np.arange(nb_class_psi), p=self.psi_law_weights)
                psi_class.append(chosen_class)
                ef_vector = np.random.multivariate_normal(psi_law_means[chosen_class], psi_law_stds[chosen_class], size=1)
                psi_graph[j, :] = ef_vector

        else:
            centre_cluster_p = np.random.randn(nb_class_alpha, self.nb_latent_factors)
            centre_cluster_d =  np.random.randn(nb_class_psi, self.nb_latent_factors)
            
            # Generate fixed effects with the gaussine sphere method
            for i in range(self.n_patients):
                chosen_class = np.random.choice(np.arange(nb_class_alpha), p=self.alpha_law_weights)
                alpha_class.append(chosen_class)
                ef_vector = np.random.multivariate_normal(centre_cluster_p[chosen_class], alpha_law_stds[chosen_class], size=1)
                ef_vector = self.radius * ef_vector/ np.linalg.norm(ef_vector)
                alpha_graph[i, :] = ef_vector
    
            for j in range(self.n_doctors):
                chosen_class = np.random.choice(np.arange(nb_class_psi), p=self.psi_law_weights)
                psi_class.append(chosen_class)
                ef_vector = np.random.multivariate_normal(centre_cluster_d[chosen_class], psi_law_stds[chosen_class], size=1)
                ef_vector = self.radius * ef_vector/ np.linalg.norm(ef_vector)
                psi_graph[j, :] = ef_vector

        # Generate coordinates and distance matrix
        for i in range(self.n_patients):
            coor_patients.append(np.random.uniform(0, 1, 2))
            for j in range(self.n_doctors):
                if i == 0:
                    coor_doctors.append(np.random.uniform(0, 1, 2))
                d = np.sqrt((coor_patients[i][0] - coor_doctors[j][0]) ** 2 + (coor_patients[i][1] - coor_doctors[j][1]) ** 2)
                D[i][j] = d
            

        # Generate continious and binary data
        sim_patient_X = rng.integers(1, 100, size=self.n_patients)
        self.sim_patient_X_normed = (sim_patient_X - sim_patient_X.mean()) / sim_patient_X.std()
        sim_doctor_X = rng.integers(1, 100, size=self.n_doctors)
        self.sim_doctor_X_normed = (sim_doctor_X - sim_doctor_X.mean()) / sim_doctor_X.std()
        sim_patient_D = np.random.choice([0, 1], self.n_patients)
        sim_doctor_D = np.random.choice([0, 1], self.n_doctors)

        # Compile IDs and features
        id_p = np.repeat(range(self.n_patients), self.n_doctors)
        id_d = np.tile(range(self.n_doctors), self.n_patients)
        X_p_data = np.repeat(sim_patient_X, self.n_doctors)
        X_d_data = np.tile(sim_doctor_X, self.n_patients)
        D_p_data = np.repeat(sim_patient_D, self.n_doctors)
        D_d_data = np.tile(sim_doctor_D, self.n_patients)
        ef_patient_data = np.repeat(alpha_graph, self.n_doctors, axis=0)
        ef_doctor_data = np.tile(psi_graph, (self.n_patients, 1))
        class_patient_data = np.repeat(alpha_class, self.n_doctors)
        class_doctor_data = np.tile(psi_class, self.n_patients)

        # Generate connection matrix
        A = np.zeros((self.n_patients, self.n_doctors))
        P = np.zeros((self.n_patients, self.n_doctors))

        # Set randomness during the link formation process 
        if self.no_seed_for_link==True:
            np.random.seed(None)
            random.seed(None)
            tf.random.set_seed(None)
        
        for i in range(self.n_patients):
            for j in range(self.n_doctors):
                if self.nb_latent_factors == 1:
                    T = alpha_graph[i] + psi_graph[j] + self.beta_X_p_graph * self.sim_patient_X_normed[i] + self.beta_X_d_graph * self.sim_doctor_X_normed[j] \
                        + self.beta_D_p_graph * sim_patient_D[i] + self.beta_D_d_graph * sim_doctor_D[j] + self.beta_distance_graph * D[i][j]
                else:
                    T = np.dot(alpha_graph[i], psi_graph[j]) + self.beta_X_p_graph * self.sim_patient_X_normed[i] + self.beta_X_d_graph * self.sim_doctor_X_normed[j] \
                        + self.beta_D_p_graph * sim_patient_D[i] + self.beta_D_d_graph * sim_doctor_D[j] + self.beta_distance_graph * D[i][j]
                p = 1 / (1 + np.exp(-T))
                P[i][j] = p
                A[i][j] = np.random.binomial(1, p)
                
        # Compile relations
        relation = A.flatten()
        dataframe = pd.DataFrame({
            'i': id_p,
            'j': id_d,
            'y': relation,
            'X_p': X_p_data,
            'X_d': X_d_data,
            'D_p': D_p_data,
            'D_d': D_d_data,
            'class_p': class_patient_data,
            'class_d': class_doctor_data
        })
        dataframe['distance'] = D[dataframe['i'], dataframe['j']].astype(float)

        # Adjust the number of connections
        number_of_connections = dataframe.groupby('i').agg({'y': 'sum'})
        zero_connection = number_of_connections[number_of_connections['y'] == 0].index
        for patient in zero_connection:
            min_index = dataframe[dataframe['i'] == patient]['distance'].idxmin()
            doctor_to_connect = dataframe.loc[min_index, 'j']
            dataframe.loc[(dataframe['i'] == patient) & (dataframe['j'] == doctor_to_connect), 'y'] = 1

        number_of_connections = dataframe.groupby('i').agg({'y': 'sum'})
        too_much_connection = number_of_connections[number_of_connections['y'] > self.max_number_connections].index
        for patient in too_much_connection:
            patient_df = dataframe[dataframe['i'] == patient]
            connected_doctors = patient_df[patient_df['y'] == 1]['j'].values
            most_popular_doctors = dataframe[dataframe['j'].isin(connected_doctors)].groupby('j').agg({'y': 'sum'}).sort_values('y', ascending=False)
            not_kept_doctors = most_popular_doctors.index[self.max_number_connections:].values
            for doctor in not_kept_doctors:
                dataframe.loc[(dataframe['i'] == patient) & (dataframe['j'] == doctor), 'y'] = 0

        # Create fixed effects dataframes and concatenate
        k = self.nb_latent_factors
        ef_patient = pd.DataFrame(np.zeros((self.n_patients * self.n_doctors, k)))
        ef_doctor = pd.DataFrame(np.zeros((self.n_patients * self.n_doctors, k)))
        for i in range(k):
            ef_patient_element = []
            ef_doctor_element = []
            ef_patient.rename(columns={i: f'ef_p_{i}'}, inplace=True)
            ef_doctor.rename(columns={i: f'ef_d_{i}'}, inplace=True)
            for j in range(self.n_patients):
                ef_patient_element += list(np.repeat(alpha_graph[j][i], self.n_doctors))
            for j in range(self.n_doctors):
                ef_doctor_element.append(psi_graph[j][i])
            ef_patient[f'ef_p_{i}'] = ef_patient_element
            ef_doctor[f'ef_d_{i}'] = np.tile(ef_doctor_element, self.n_patients)
        dataframe = pd.concat([dataframe, ef_patient, ef_doctor], axis=1)
        dataframe = dataframe.reset_index().drop(['index'], axis=1)

        density = dataframe[dataframe['y']==1].shape[0]/dataframe.shape[0]

        # Assigning results to attributes
        self.df = dataframe
        self.alpha_graph = alpha_graph
        self.psi_graph = psi_graph
        self.alpha_class = alpha_class
        self.psi_class = psi_class
        self.coor_patients = coor_patients
        self.coor_doctors = coor_doctors
        self.D = D
        self.sim_patient_D= sim_patient_D
        self.sim_doctor_D = sim_doctor_D
        self.density = density
        self.P = P

        if show_time_execution == True:
            end_time = time.time()
            print(f"Temps d'exécution pour construire le graphe : {end_time - start_time:.0f} secondes")

