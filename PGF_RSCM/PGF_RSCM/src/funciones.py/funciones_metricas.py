#Métricas 

# Librerías -------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import math
import matplotlib.pyplot as plt
import scikitplot as skplt
import pickle

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from functools import partial
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, auc

# Funciones -------------------------------------------------------------------------------------------------------------------------------

# Matriz de confusión normalizada y sin normalizar 
def plot_confusion_matrices(y_train, y_pred, class_labels=['No Fraude', 'Fraude']):
    """
    Plotea la matriz de confusión y la matriz de confusión normalizada.

    Parameters:
    - y_true: Etiquetas reales.
    - y_pred: Etiquetas predichas.

    Returns:
    - None
    """
    # Calcular la matriz de confusión
    confusion = confusion_matrix(y_train, y_pred)

    # Plotear la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.title('Matriz de Confusión Valor Absolutos')
    plt.show()

    # Normalizar la matriz de confusión
    confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

    # Plotear la matriz de confusión normalizada con porcentajes
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.title('Matriz de Confusión Normalizada')
    plt.show()
    
    
    
# Accuracy score 
def calcular_accuracy_score(y_train, y_pred):
    """
    Calcula y devuelve la precisión (accuracy score).

    Returns:
    - Precisión (accuracy score).
    """
    accuracy = accuracy_score(y_train, y_pred)
    return accuracy



# Curva ROC
def plot_roc_curve(y_train, y_pred):
    fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Modelo')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('Curva ROC')
    plt.show()

    
    
#Métricas: precision, recall, f1 score 
def evaluar_modelo_precision_recall(y_train, y_pred):
    """
    Evalúa un modelo de clasificación binaria utilizando la curva Precision-Recall y F1-score.

    Retorna:
    - Umbral óptimo, F1-score y visualización de la curva Precision-Recall.
    """
    # Calcula la curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_train, y_pred)

    # Calcula el F1-score
    fscore = (2 * precision * recall) / (precision + recall)

    # Encuentra el índice del umbral óptimo para el F1-score
    ix = np.argmax(fscore)

    # Muestra el umbral óptimo y el F1-score correspondiente
    print(f'Best Threshold={thresholds[ix]:.3f}, F1-Score={fscore[ix]:.3f}')

    # Dibuja la curva Precision-Recall y señala el mejor punto
    no_skill = len(y_train[y_train == 1]) / len(y_pred)
    plt.plot([0, 1], [no_skill, no_skill])
    plt.plot(recall, precision, marker='.', label='Modelo')
    plt.scatter(recall[ix], precision[ix], s=100, marker='o', color='black', label='Best')

    # Etiquetas y leyenda del gráfico
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # Muestra el gráfico
    plt.show()

    # Calcula y muestra el F1-score usando la función de scikit-learn
    score = f1_score(y_train, y_pred)
    print(f'F-Score: {score:.5f}')

    # Retorna el umbral óptimo y el F1-score
    return thresholds[ix], fscore[ix]



# Curva acumulativa 
def plot_cumulative_gain_curve(y_train, y_pred):
    """
    Plotea la curva de ganancia acumulativa.

    Parameters:
    - y_true: Etiquetas reales.
    - predicted_probas: Probabilidades predichas para las etiquetas positivas.

    Returns:
    - None
    """
    skplt.metrics.plot_cumulative_gain(y_train, y_pred)
    plt.show()
    
    
    
# Curva Lift  
def plot_lift_curve(y_train, y_pred):
    """
    Plotea la curva Lift.

    Parámetros:
    - y_true: Etiquetas reales del conjunto de prueba.
    - y_prob: Probabilidades predichas para las etiquetas positivas.
    """
    skplt.metrics.plot_lift_curve(y_train, y_pred)
    plt.show()
    
    
    
# Partial Dependence Plots 
def calculate_pdp_metric(model, X, feature_index, feature_values):
    """
    Calcula la métrica de Partial Dependence Plot para una característica específica.

    Parameters:
    - model: Modelo entrenado (por ejemplo, GradientBoostingClassifier).
    - X: Conjunto de datos de características.
    - feature_index: Índice de la característica de interés.
    - feature_values: Valores de la característica para los cuales calcular la métrica.

    Returns:
    - pdp_metric: Métrica del Partial Dependence Plot.
    """
    # Ajustar el codificador One-Hot a las características categóricas si es necesario
    try:
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)
    except ValueError:
        X_encoded = X

    # Calcular el PDP usando el modelo y la característica de interés
    pdp, axes = partial_dependence(model, X_encoded, features=[feature_index])

    # Calcular la métrica del PDP para los valores de la característica específica
    pdp_metric = np.interp(feature_values, axes[0], pdp[0])

    return pdp_metric

def plot_pdp_metric(feature_values, pdp_metric, feature_name):
    """
    Plotea la métrica del Partial Dependence Plot para una característica específica.

    Parameters:
    - feature_values: Valores de la característica.
    - pdp_metric: Métrica del Partial Dependence Plot.
    - feature_name: Nombre de la característica.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(feature_values, pdp_metric, marker='o')
    plt.title(f'Partial Dependence Plot Metric for {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Partial Dependence Metric')
    plt.grid(True)
    plt.show()
    
    
# Pipeline
def Pipeline(classifier):
    # Crea el pipeline con el clasificador
    pipe = Pipeline(
        ('classifier', classifier)
    )
    
    # Muestra los resultados
    print(f"Classifier: {classifier.__class__.__name__}")
    print("Model score: %.3f" % accuracy_score(y_val, y_pred))
    print("\n")

# Métrica final empleada 
def custom_metrics(y_train, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return sensitivity - 0.1 * specificity 
    
# Pickle    
def guardar_modelo(modelo, nombre_archivo='modelo.pkl'):
    # Guarda el modelo en un archivo pickle
    with open(nombre_archivo, 'wb') as file:
        pickle.dump(modelo, file)
    print(f"Modelo guardado en '{nombre_archivo}'")

def cargar_modelo(nombre_archivo='modelo.pkl'):
    # Carga el modelo desde un archivo pickle
    with open(nombre_archivo, 'rb') as file:
        modelo_cargado = pickle.load(file)
    print(f"Modelo cargado desde '{nombre_archivo}'")
    return modelo_cargado