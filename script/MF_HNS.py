### SCRIPT pour l'entrainement des modèles ###

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import time

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

import script.decorateur as decorateur
import script.graph_formation as graph_formation

GraphFormation = graph_formation.GraphFormation
#tf.compat.v1.config.run_functions_eagerly(True)        
#tf.compat.v1.disable_eager_execution()

# Pour arrêter la descente lorsque la loss est suffisamment faible
class StopTrainingBelowLoss(Callback):
    def __init__(self, target_loss):
        super(StopTrainingBelowLoss, self).__init__()
        self.target_loss = target_loss

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') is not None and logs.get('loss') < self.target_loss:
            print(f"\nTraining stopped as the loss reached below {self.target_loss}")
            self.model.stop_training = True

# Récupérer la valeur de la loss (utile si on ne fixe pas de seuil à atteindre)
class LossHistory(Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))

def get_estimations(df, nb_epochs=50, dim_embedding=1, initial_weights=None, target_loss=None, show_print=0, seed=12, l_lambda=0, regularization="l2", show_time_execution = False):
    """Etant donné un dataframe, get_estimations renvoie l'estimation des effets fixes et des Bêtas en utilisant TensorFlow.keras

    Args:
        df (_type_): le dataframe pour lequel on souhaite obtenir l'estimation
        nb_epochs (int, optional): Le nombre d'itérations pour notre descente de gradient. Defaults to 50.
        initial_weights (_type_, optional): Si on a déjà lancé un entraînement et qu'on souhaite le poursuivre, on peut rentrer l'estimation obtenue auparavant
        (voir plus bas pour un exemple). Defaults to None.
        target_loss (_type_, optional): Si une valeur est indiquée, la descente de gradient ne s'arrête pas tant que la loss n'est pas inférieure
        à cette valeur. Defaults to None.

    Returns:
        _type_: l'ensemble des estimations (voir plus bas pour un exemple)
    """
    start_time = time.time()
    # fixation des seeds pour être reproductible
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Définition des dimensions
    num_patients = df['i'].nunique()
    num_doctors = df['j'].nunique()

    # Crée un LabelEncoder pour les ID des patients et des docteurs, utile si les id ne sont pas nécessairement une suite consécutive d'entiers (1, 2, ..., n)
    patient_encoder = LabelEncoder()
    i_encoded = patient_encoder.fit_transform(df['i'])

    doctor_encoder = LabelEncoder()
    j_encoded = doctor_encoder.fit_transform(df['j'])

    # Enregistrer les mappings ID -> entier pour retrouver les bons indices associés à chaque patient/docteur si besoin
    patient_id_mapping = dict(zip(patient_encoder.classes_, range(num_patients)))
    doctor_id_mapping = dict(zip(doctor_encoder.classes_, range(num_doctors)))

    y = df['y'].values
    df = df.astype(np.float32)

    # Renormalisation des variables X (comme lors de la génération)
    df['X_p'] = ( df['X_p'] -  df['X_p'].mean())/ df['X_p'].std()
    df['X_d'] = ( df['X_d'] -  df['X_d'].mean())/  df['X_d'].std()
    
    X = [i_encoded, j_encoded, df['X_p'].values, df['X_d'].values, df['D_p'].values,\
            df['D_d'].values, df['distance'].values]
    
    # Entrées du modèle
    user_input = Input(shape=(1,), name="i")
    doctor_input = Input(shape=(1,), name="j")
    X_patient_input = Input(shape=(1,), name='X_p')
    X_doctor_input = Input(shape=(1,), name='X_d')
    D_patient_input = Input(shape=(1,), name='D_p')
    D_doctor_input = Input(shape=(1,), name='D_d')
    distance_input = Input(shape=(1,), name="distance")

    # Incorporation des utilisateurs et des docteurs dans des espaces latents
    if regularization == "l1":
        
        user_embedding = Embedding(name = 'patient_embedding', input_dim=num_patients, output_dim=dim_embedding, embeddings_regularizer=regularizers.l1(l_lambda))(user_input)
        doctor_embedding = Embedding(name = 'doctor_embedding', input_dim=num_doctors, output_dim=dim_embedding, embeddings_regularizer=regularizers.l1(l_lambda))(doctor_input)
    elif regularization == "l2":
        user_embedding = Embedding(name = 'patient_embedding', input_dim=num_patients, output_dim=dim_embedding, embeddings_regularizer=regularizers.l2(l_lambda))(user_input)
        doctor_embedding = Embedding(name = 'doctor_embedding', input_dim=num_doctors, output_dim=dim_embedding, embeddings_regularizer=regularizers.l2(l_lambda))(doctor_input)
    else:
        raise ValueError("the regularization parameter must be 'l1' or 'l2'")
    
    # Obtention des vecteurs latents des utilisateurs et des docteurs
    user_latent = Flatten()(user_embedding)
    doctor_latent = Flatten()(doctor_embedding)

    # Produit scalaire entre les vecteurs latents des utilisateurs et des docteurs
    #dot_product = Dot(axes=1)([user_latent, doctor_latent])

    # Création d'une couche pour paramétriser les Beta
    class CustomLayer(Layer):
        def __init__(self, **kwargs):
            super(CustomLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.beta_X_patient = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)
            self.beta_X_doctor = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)
            self.beta_D_patient = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)
            self.beta_D_doctor = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)
            self.beta_distance = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)
            super(CustomLayer, self).build(input_shape)

        def call(self, inputs):
            X_patient_input, X_doctor_input, D_patient_input, D_doctor_input, distance_input = inputs
            linear_term = self.beta_X_patient * X_patient_input + \
                        self.beta_X_doctor * X_doctor_input + \
                        self.beta_D_patient * D_patient_input + \
                        self.beta_D_doctor * D_doctor_input + \
                        self.beta_distance * distance_input
            return linear_term

    # Ajout de la couche personnalisée dans le modèle
    linear_term = CustomLayer()([X_patient_input, X_doctor_input, D_patient_input, D_doctor_input, distance_input])  # X*beta

    if dim_embedding == 1:
        output = Add()([user_latent, doctor_latent, linear_term]) # X*beta + psi_i + alpha_j
    else:
         # Produit scalaire entre les vecteurs latents des utilisateurs et des docteurs
        dot_product = Dot(axes=1)([user_latent, doctor_latent]) 
        output = Add()([dot_product, linear_term])
    output = Activation('sigmoid')(output)  # sigma(.)

    # Création du modèle
    model = Model(inputs=[user_input, doctor_input, X_patient_input, X_doctor_input, D_patient_input, D_doctor_input, distance_input], outputs=output)

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Chargement des poids initiaux si disponibles (utile pour reprendre l'entraînement)
    if initial_weights != None:
        model.set_weights(initial_weights)

    # Ajout du rappel pour arrêter l'entraînement seulement si la perte atteint un seuil
    if target_loss != None:
        callbacks = []
        callbacks.append(StopTrainingBelowLoss(target_loss))
        # Initialisation du nombre d'époques
        epoch = 0
    # Boucle d'entraînement jusqu'à ce que la perte atteigne le seuil
        while True:
            # Entraînement d'une époque
            history = model.fit(X, y, epochs=1, batch_size=64, validation_split=0.2, verbose=show_print, callbacks=callbacks)

            # Incrémentation du nombre d'époques
            epoch += 1

            # Vérification si l'entraînement doit être arrêté
            if model.stop_training:
                break
        return epoch, model.get_weights(), patient_id_mapping, doctor_id_mapping, model
    else:
        # Ajout du rappel pour enregistrer la perte
        loss_history = LossHistory()
        callbacks = [loss_history]
        # Entraînement du modèle
        #model.fit(X, y, epochs=nb_epochs, batch_size=64, validation_split=0.2, verbose=show_print, callbacks=callbacks)
        history = model.fit(X, y, epochs=nb_epochs, batch_size=64, validation_split=0.2, verbose=show_print, callbacks=callbacks) # Pour ne pas afficher le print des epochs
        loss_values = loss_history.losses

        if show_time_execution == True:
            end_time = time.time()
            print(f"temps d'execution pour l'entrainement: {end_time - start_time:.0f}")
        
        return loss_values, model.get_weights(), patient_id_mapping, doctor_id_mapping, model, history


@decorateur.compute_time
def prediction_score(graph_object, nb_epochs = 100, train_test_split = 0.8, seed = 12, regularization = "l2", l_lambda= 1e-7, initial_weights=None, target_loss=None, show_print=0):
    """
    Calculer et évaluer le score de prédiction d'un modèle sur les données d'un objet graph_object.

    Paramètres :
    -----------
    graph_object : objet
        Un objet contenant le DataFrame `df` et les paramètres du modèle.
    nb_epochs : int, optionnel
        Le nombre d'époques pour entraîner le modèle, par défaut 100.
    train_test_split : float, optionnel
        La proportion du jeu de données à inclure dans l'ensemble d'entraînement, par défaut 0.8.
    seed : int, optionnel
        Graine aléatoire pour la reproductibilité, par défaut 12.
    regularization : str, optionnel
        Type de régularisation à utiliser ("l2", "l1", etc.), par défaut "l2".
    l_lambda : float, optionnel
        Paramètre de régularisation, par défaut 1e-7.
    initial_weights : liste ou None, optionnel
        Poids initiaux pour le modèle, par défaut None.
    target_loss : float ou None, optionnel
        Perte cible pour l'arrêt anticipé, par défaut None.
    show_print : int, optionnel
        Affiche ou non les loss durant les epochs

    Retourne :
    ---------
    None

    Description :
    ------------
    Cette fonction divise les données en ensembles d'entraînement et de test, 
    entraîne un modèle sur l'ensemble d'entraînement et évalue sa performance sur l'ensemble de test.
    Elle trace l'historique de l'entraînement, la matrice de confusion et imprime diverses métriques de performance
    telles que la précision, le rappel, le F1-score et l'erreur quadratique moyenne.

    La fonction imprime également les coefficients estimés (bêtas) et leurs erreurs par rapport
    aux paramètres réels de l'objet graph_object.

    Exemple :
    --------
    >>> graph = GraphFormation(n_patients=100, n_doctors=50, max_number_connections=10)
    >>> graph.do_the_graph()
    >>> prediction_score(graph, nb_epochs=200, train_test_split=0.8)

    """
    
    # On prend 80% du dataframe
    train_indices = int(graph_object.df.shape[0]*train_test_split)
    # L'argument drop=True permet de ne pas ajouter l'ancien index comme colonne supplémentaire dans le DataFrame
    df_shuffled = graph_object.df.sample(frac=1).reset_index(drop=True)
    df_shuffled['X_p'] = (df_shuffled['X_p'] - df_shuffled['X_p'].mean())/df_shuffled['X_p'].std()
    df_shuffled['X_d'] = (df_shuffled['X_d'] - df_shuffled['X_d'].mean())/df_shuffled['X_d'].std()

    df_train ,df_test = df_shuffled.iloc[:train_indices,:] , df_shuffled.iloc[train_indices:,:]

    # Training sur les data d'entrainement
    estimates = get_estimations(df_train, nb_epochs=nb_epochs, initial_weights=initial_weights, target_loss=target_loss, show_print=show_print, seed=seed,      regularization=regularization, l_lambda= l_lambda)
    model = estimates[4] 
    # Récupération de la loss
    history = estimates[5]
    loss = estimates[0]

    # Créer une figure et des sous-graphiques (axes)
    fig, (loss_plot) = plt.subplots(1, 1, figsize=(12, 8))  
    loss_plot.plot(history.history["loss"])
    loss_plot.plot(history.history["val_loss"]) 
    loss_plot.set_title("model loss")
    loss_plot.set_ylabel("loss")
    loss_plot.set_xlabel("epoch")
    loss_plot.legend(["train", "val"], loc="upper left")
    # Trouver le minimum de la courbe de validation
    min_val_loss = min(history.history["val_loss"])
    min_val_loss_epoch = history.history["val_loss"].index(min_val_loss)
    
    # Ajouter un point rouge au minimum de la courbe de validation
    loss_plot.plot(min_val_loss_epoch, min_val_loss, 'ro')
    
    # Afficher les informations du point minimum
    loss_plot.text(min_val_loss_epoch, min_val_loss, f'Min: {min_val_loss:.4f}', color='red', fontsize=12)

    # Drop unwanted columns
    X_test = df_test.drop(['y', 'class_p', 'class_d', 'ef_p_0', 'ef_d_0',], axis=1)
    
    # Pour pouvoir les rentrer dans le modèle
    input_data = [tf.constant(X_test[col].values.reshape(-1, 1), dtype=tf.float32) for col in X_test.columns]
    
    # Predict on test data
    results = model.predict(input_data)
    y_true = df_test['y'].values.astype(int)
    y_pred =  (results.flatten() > 0.5).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mse = np.sum((results.flatten()-df_test['y'])**2)/1500

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    # Affiche la matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix on test dataframe")
    #confusion_matrix_plot.plot(cmap=plt.cm.Blues)
    #confusion_matrix_plot.set_title("Confusion matrix on test dataframe")

    # Affiche divers métrique
    print(f"Mean Squared Error sur l'échantillon de test : {mse}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

    beta_X_p = estimates[1][2]
    beta_X_d = estimates[1][3]
    beta_D_p = estimates[1][4]
    beta_D_d = estimates[1][5]
    beta_distance = estimates[1][6]
    
    print(f"ESTIMATION BETA: X p {beta_X_p}; X d {beta_X_d}; D_p {beta_D_p}; D_d {beta_D_d}; distance {beta_distance}")
    print(f"ERROR ESTIMATION BETA: X p {beta_X_p-graph_object.beta_X_p_graph}; X d {beta_X_d-graph_object.beta_X_d_graph}; D_p {beta_D_p-graph_object.beta_D_p_graph}; D_d {beta_D_d-graph_object.beta_D_d_graph}; distance {beta_distance - graph_object.beta_distance_graph}")
    plt.show()

@decorateur.compute_time
def loss_vs_density(sim_beta_distance_array = [-20,-15, -12, -10,-7,-5, -3, -2,], nb_epochs = 120):
    """
    Évalue la perte en fonction de la densité du graphe pour différentes valeurs de beta_distance.

    Paramètres :
    -----------
    sim_beta_distance_array : list, optionnel
        Liste des valeurs de beta_distance pour lesquelles le graphe sera simulé.
        Par défaut, [-20, -15, -12, -10, -7, -5, -3, -2].
    nb_epochs : int, optionnel
        Le nombre d'époques pour l'entraînement du modèle. Par défaut à 120.

    Description :
    ------------
    Cette fonction simule des graphes pour chaque valeur de beta_distance dans `sim_beta_distance_array`.
    Pour chaque graphe simulé, elle calcule la densité du graphe, affiche cette densité, et entraîne un modèle
    avec les données du graphe en utilisant `prediction_score`. Cela permet de comparer les courbes de loss et les minimums.

    Exemple :
    --------
    loss_vs_density(sim_beta_distance_array=[-15, -10, -5], nb_epochs=100)
    """
    for sim_beta_distance in sim_beta_distance_array:
        graph_object= GraphFormation(
                    n_patients=1000,
                     n_doctors=50,
                     max_number_connections=50,
                    beta_distance_graph = sim_beta_distance)
        graph_object.do_the_graph()
        print(f"density of the graph: {graph_object.density*100:.2f}%")
        prediction_score(graph_object, nb_epochs=nb_epochs)

def hard_negative_sampling_first_step(dataframe,
                                      nb_Y = 1,
                                     ):
    """
    A partir de notre dataframe, on doit pouvoir générer des Y = 0 pertinents. Pour cela, il faut effectuer un entrainement initial sur notre dataframe pour 
    avoir une première estimation des paramètres de notre modèle (EF, Betas) et ensuite effectuer ce HNS.

    Args:
        dataframe (_type_): Le dataframe initial sur lequel on applique la première étape de notre Hard Negative Sampling.
        nb_Y (int, optional): Le nombre de Y=0 que l'on souhaite échantillonner. Defaults to 1.

    Returns:
        dataframe: Renvoie le dataframe échantillonné.
    """
    df = dataframe.copy()
    # Hard Negative Sampling : 

    # On enregistre l'ensemble des Y = 0 avant de les supprimer de notre dataframe
    negative_connections = df[df['y'] == 0]
    negative_patients = negative_connections['i'].unique()
    # On supprime les Y = 0 car on va les générer avec le HNS
    df.drop(df[df['y'] == 0].index, inplace = True)
    df = df.reset_index(drop=True)
    # D'abord, on effectue l'étape 1 du HNS. Il nous faut une première estimation des EFs/Betas. On garde un Y=0 en se basant sur une distribution dépendant de la
    # popularité/distance à chaque docteur. Pour chaque patient, on garde en mémoire le(s) Y=0 le plus intéressant.
    negative_connections_first_step = negative_connections.copy()
    for i in negative_patients:
        patient_df = negative_connections_first_step[negative_connections_first_step['i'] == i]
        distance = np.zeros(len(patient_df)) # Array pour stocker les distances. Il faut faire attention, indice de l'array =/= indice "réel" du docteur
        popularity = np.zeros(len(patient_df)) # Array pour stocker les popularités
        for j, doctor in enumerate(patient_df['j']):
            distance[j] = patient_df[patient_df['j'] == doctor]['distance'].iloc[0]
            popularity[j] = df[df['j'] == doctor]['y'].sum()
        score = popularity / distance
        score = score/score.sum()
        
        # On conserve les échantillons négatifs les plus significatifs (ceux pour lesquels la probabilité de connexion est la plus grande mais n'a pas eu lieu)
        best_score_array = score.argsort()[-nb_Y:]
        doctors_chosen = patient_df['j'].iloc[best_score_array].to_list()
        doctors_to_eliminate = patient_df[patient_df['j'].isin([j for j in patient_df['j'] if j not in doctors_chosen])]
        negative_connections_first_step.drop(doctors_to_eliminate.index, inplace = True)
    # On a maintenant le Y = 0 désiré pour chaque patient, il faut fusionner les Y = 0 et Y = 1
    return pd.concat([df, negative_connections_first_step]).reset_index(drop=True)


def HNS_and_estimates(dataframe,
                      M,
                      nb_Y = 1,
                      epochs=50,
                      weights=None,
                      threshold_loss=None,
                      show_print=0,
                      regularization = 'l2',
                      l_lambda=1e-7
                      ):
    """
    Cette fonction effectue le Hard Negative Sampling (HNS) dans sa globalité et renvoie l'estimation finale des effets fixes, betas.

    Args:
        dataframe (_type_): Le dataframe initial
        M (_type_): Le nombre associé au Top-M docteurs (le paramètre de notre HNS permettant de tirer aléatoirement parmi
        les M docteurs les plus relevants, cf. papier de recherche associé)
        nb_Y (int, optional): Le paramètre Y associé à la première étape du HNS. Defaults to 1.
        epochs (int, optional): paramètre de get_estimations. Defaults to 50.
        weights (_type_, optional): paramètre de get_estimations. Defaults to None.
        threshold_loss (float, optional): paramètre de get_estimations. Defaults to 0.01.

    Returns:
        _type_: l'ensemble des estimations.
    """
    df = dataframe.copy()
    n_patients = dataframe['i'].nunique()
    n_doctors = dataframe['j'].nunique()
    # On récupère le dataframe ayant subi la première étape du HNS
    df_first_hns = hard_negative_sampling_first_step(df, nb_Y)
    
    # On utilise get_first_estimations qui s'occupe d'effectuer la première estimation des EF/Bêtas.
    parameters = get_estimations(df_first_hns, l_lambda = l_lambda, regularization = regularization, nb_epochs=epochs, initial_weights=weights, target_loss=threshold_loss, show_print=show_print)

    # parameters[2] contient l'ensemble des premières estimations

    # Maintenant, on estime les scores en se servant de la première estimation ci-dessus (https://arxiv.org/abs/2302.03472)
    alpha_graph_training = parameters[1][0]
    psi_graph_training = parameters[1][1]
    beta_X_p_graph_training = parameters[1][2]
    beta_X_d_graph_training = parameters[1][3]
    beta_D_p_graph_training = parameters[1][4]
    beta_D_d_graph_training = parameters[1][5]
    beta_distance_graph_training =  parameters[1][6]
    prediction_scores = np.zeros((n_patients, n_doctors))
    negative_drawing_list = []
    # Les scores estimés se basent sur notre modèle de LMF (https://arxiv.org/abs/2302.03472)
    negative_connections = df[df['y'] == 0]
    negative_patients = negative_connections['i'].unique()
    highest_prediction_scores_indexes = np.zeros((len(negative_patients), M))
    # Pour calculer les scores, il faut au préalable récupérer les âges et sexes normalisés.

    df['X_p_normed'] = ( df['X_p'] - df['X_p'].mean() ) / df['X_p'].std()
    df['X_d_normed'] = ( df['X_d'] - df['X_d'].mean() ) / df['X_d'].std()

    print(alpha_graph_training.shape, psi_graph_training.shape, beta_X_p_graph_training, beta_X_d_graph_training, beta_D_p_graph_training, beta_D_d_graph_training,  beta_distance_graph_training)
    
    for i in negative_patients:
        patient_df = negative_connections[negative_connections['i'] == i]
        patient_X_normed = df[df['i'] == i]['X_p_normed'].iloc[0]
        patient_D_normed = df[df['i'] == i]['D_p'].iloc[0]
        # On considère les docteurs pour lesquels le patient i n'a pas de connexions avec
        for j in patient_df['j']:
            if j>=psi_graph_training.shape[0]:
                break
            # on récupère la distance entre le patient i et le docteur j
            distance = patient_df[patient_df['j'] == j]['distance'].iloc[0]
            doctor_X_normed = df[df['j'] == j]['X_d_normed'].iloc[0]
            doctor_D_normed =df[df['j'] == j]['D_d'].iloc[0]

            T = alpha_graph_training[i] + psi_graph_training[j] + beta_X_p_graph_training * patient_X_normed + beta_X_d_graph_training * doctor_X_normed \
            + beta_D_p_graph_training * patient_D_normed + beta_D_d_graph_training * doctor_D_normed + beta_distance_graph_training * distance
            prediction_scores[i][j] = 1 / (1 + np.exp(-T))
        # on garde les M indices des docteurs avec la plus grande probabilité de connexion pour le patient i dans highest_prediction_scores_indexes
        highest_prediction_scores_indexes[i] = np.argsort(prediction_scores[i])[-M:]
        # on effectue le negative sampling
        negative_drawing_patient = []
        # on détermine les y = 0 à rajouter à notre dataframe
        negative_drawing = np.random.binomial(1, p=1/M, size=M)
        for l in range(M):
            if negative_drawing[l] == 1:
                doc = highest_prediction_scores_indexes[i][l]
                negative_drawing_patient.append(patient_df[patient_df['j'] == doc].iloc[0].to_list())
        negative_drawing_list += negative_drawing_patient
    # Une fois les connexions négatives récupérées pour chaque patient, on concatène nos Y = 1 et Y = 0
    positive_connections = dataframe.drop(dataframe[dataframe['y'] == 0].index)
    positive_connections = positive_connections.reset_index(drop=True)
    df_second_hns = pd.DataFrame(negative_drawing_list)
    df_second_hns.columns = dataframe.columns
    df_second_hns = pd.concat([positive_connections, df_second_hns], axis = 0)
    df_second_hns = df_second_hns.reset_index(drop=True)
    
    # On effectue maintenant l'estimation finale des EFs/Betas encore une fois à l'aide de Keras mais cette fois-ci sur notre second dataframe
    final_parameters = get_estimations(df_second_hns, l_lambda= l_lambda, regularization=regularization, nb_epochs=epochs, initial_weights=weights, target_loss=threshold_loss, show_print=show_print)

    return final_parameters
