# ==============================================================================
# FUNCIONES PARA PRÁCTICAS
# ==============================================================================
# LIBRERÍAS
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import math
from functools import partial
# ==============================================================================

# VALORES MISSING TRATADOS DATASET
def analyze_missing_data(df, variables):
    missing_count = []
    missing_percentage = []
    fraud_count = []  
    fraud_percentage = [] 

    for variable in variables:
        if variable == 'intended_balcon_amount':
            missing_count.append(df[df[variable] < 0].shape[0])
        else:
            missing_count.append(df[df[variable] == -1].shape[0])

        missing_percentage.append((missing_count[-1] / len(df)) * 100)

        if missing_count[-1] > 0:
            fraud_count.append(df[(df[variable] == -1) & (df['fraud_bool'] == 1)].shape[0])
            fraud_percentage.append((fraud_count[-1] / missing_count[-1]) * 100)
        else:
            fraud_count.append(0)
            fraud_percentage.append(0)

    missing_data_df = pd.DataFrame({
        'Variable': variables,
        'Missing': missing_count,
        'Porcentaje Missing': missing_percentage,
        'Numero de 1 en Nulos': fraud_count,
        'Porcentaje de 1 en Nulos': fraud_percentage
    })

    return missing_data_df

# MATRIZ DE CONFUSIÓN ABSOLUTO
def calcular_matriz_confusion_absoluto(y_true, y_pred):
    matriz_confusion = confusion_matrix(y_true, y_pred)
    return matriz_confusion

# MATRIZ DE CONFUSIÓN RELATIVO
def calcular_matriz_confusion_relativo(y_true, y_pred):
    matriz_confusion = confusion_matrix(y_true, y_pred)
    total_clases_reales = np.sum(matriz_confusion, axis=1, keepdims=True)
    matriz_confusion_relativa = matriz_confusion / total_clases_reales.astype(float)
    return matriz_confusion_relativa

# VISUALIZACIÓN MATRIZ DE CONFUSIÓN 
def visualizacion_matriz_confusion(matriz_confusion, class_labels):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(matriz_confusion, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.title('Matriz de Confusión Valor Absolutos')
    plt.show()

#IDENTIFICACION OUTLIERS DE VARIABLES CONTINUAS
def outliers(pd_loan, list_var_continuous, target, multiplier):
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size / size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size / size_s
        
        if perc_excess > 0:
            # Filtrar los datos que están fuera del intervalo de confianza
            outliers = pd_loan[(pd_loan[i] < left) | (pd_loan[i] > right)]
            
            # Calcular el porcentaje de fraud_bool=1 y fraud_bool=0 en los outliers
            perc_fraud_1 = outliers[target].value_counts(normalize=True).get(1, 0)
            perc_fraud_0 = outliers[target].value_counts(normalize=True).get(0, 0)
            
            # Redondear hacia arriba el número de outliers totales
            total_outliers = math.ceil(perc_excess * size_s)
            
            pd_concat_percent = pd.DataFrame({
                'variable': [i],
                'sum_outlier_values_custom': [total_outliers],
                'porcentaje_outlier_values': [perc_excess],
                'perc_fraud_1': [perc_fraud_1],
                'perc_fraud_0': [perc_fraud_0]
            })
            
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

#CATEGORIZA COLUMNAS
def categorizar_columnas(df):
    columnas_categoricas = []
    columnas_continuas = []

    for columna in df.columns:
        # Excluir la columna 'fraud_bool' de ambas listas
        if columna == 'fraud_bool':
            continue

        # Verificar si la columna es de tipo objeto (categórica)
        if df[columna].dtype == 'object':
            print(f"Conteo de valores para '{columna}':")
            print(df[columna].value_counts())
            print("\n")
            columnas_categoricas.append(columna)
        else:
            columnas_continuas.append(columna)
            
    return columnas_categoricas, columnas_continuas

#VISUALIZACIÓN DE VARIABLES CATEGORICAS 
def plot_categorical_feature(df, target):
    """
    Visualize categorical variables with and without faceting on the target variable.
    
    Parameters:
    - df: DataFrame
    - target: str, the target variable
    
    Returns:
    - None
    """
    for col_name in df.columns:
        f, ax1 = plt.subplots(figsize=(12, 6), dpi=90)
        
        count_null = df[col_name].isnull().sum()

        sns.countplot(x=col_name, hue=target, data=df, palette=['#5975A4', '#FFA07A'], saturation=1, ax=ax1)
        ax1.tick_params(axis='x', rotation=90)
        ax1.set_xlabel(col_name, fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'{col_name} - Numero de nulos: {count_null}', fontsize=14)
        ax1.legend(title=target)
        
        plt.tight_layout()
        
#VISUALIZACIÓN DE VARIABLES NUMERICAS ENTERAS     
def plot_numeric_int_feature(df, target):
    """
    Visualize categorical variables with and without faceting on the target variable.
    
    Parameters:
    - df: DataFrame
    - target: str, the target variable
    
    Returns:
    - None
    """
    int_columns = df.select_dtypes(include='int64').columns

    for col_name in int_columns:
        f, ax1 = plt.subplots(figsize=(12, 6), dpi=90)

        count_null = df[col_name].isnull().sum()

        sns.countplot(x=col_name, hue=target, data=df, palette=['#5975A4', '#FFA07A'], saturation=1, ax=ax1)
        ax1.tick_params(axis='x', rotation=90)
        ax1.set_xlabel(col_name, fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'{col_name} - Numero de nulos: {count_null}', fontsize=14)
        ax1.legend(title=target)

        plt.tight_layout()
        
#VISUALIZACIÓN DE VARIABLES NUMERICAS CONTINUAS
def plot_numeric_feature(df, col_name, target):
    """
    Visualize a numeric variable with and without faceting on the target variable.
    
    Parameters:
    - df: DataFrame
    - col_name: str, the column to visualize
    - target: str, the target variable
    
    Returns:
    - None
    """
    f, ax2 = plt.subplots(figsize=(8, 3), dpi=90)
        
    count_null = df[col_name].isnull().sum()

    sns.boxplot(x=target, y=col_name, data=df, palette=['#5975A4', '#FFA07A'], ax=ax2)
    ax2.set_ylabel(col_name)
    ax2.set_xlabel('')
    ax2.set_title(f'{col_name} by {target}')

    plt.tight_layout()

    
# MATRIZ DE CORRELACIÓN PARA VARIABLES CONTINUAS
def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return corr

# MATRIZ DE CONFUSIÓN VALORES ÚNICOS
def show_lower_triangle_values(corr_matrix):
    # Obtén la parte inferior de la matriz de correlación
    lower_triangle = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
    
    # Filtra los valores no nulos
    lower_values = lower_triangle.stack()[lower_triangle.stack() != 0]

    # Convierte los resultados en un DataFrame
    lower_df = pd.DataFrame({'Variable 1': lower_values.index.get_level_values(0),
                             'Variable 2': lower_values.index.get_level_values(1),
                             'Correlation': lower_values.values})


    lower_df = lower_df.sort_values(by='Correlation', key=lambda x: np.abs(x), ascending=False)

    return lower_df
    
# MATRIZ DE CORRELACIÓN PARA VARIABLES CATEGÓRICAS
def cramer_matrix(df, columnas_categoricas):
    cramer_matrix = np.zeros((len(columnas_categoricas), len(columnas_categoricas)))

    for i, col1 in enumerate(columnas_categoricas):
        for j, col2 in enumerate(columnas_categoricas):
            if i != j:
                contingency_table = pd.crosstab(df[col1], df[col2])
                cramers_value = cramers_v(contingency_table.values)
                cramer_matrix[i, j] = cramers_value

    new_corr_df = pd.DataFrame(cramer_matrix, index=columnas_categoricas, columns=columnas_categoricas)
    new_corr_df = new_corr_df.abs()
    new_corr_df.loc[:, :] = np.tril(new_corr_df, k=-1)
    new_corr_df = new_corr_df.stack().to_frame('correlation').reset_index().sort_values(by='correlation', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(new_corr_df.pivot(index='level_0', columns='level_1', values='correlation'), 
                annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True)
    plt.title("Matriz de Correlación (Cramers V)")
    plt.show()
    
# FUNCIONES PARA CRAMER
def table_cramer_matrix(df, columnas_categoricas):
    cramer_matrix = np.zeros((len(columnas_categoricas), len(columnas_categoricas)))

    for i, col1 in enumerate(columnas_categoricas):
        for j, col2 in enumerate(columnas_categoricas):
            if i != j:
                contingency_table = pd.crosstab(df[col1], df[col2])
                cramers_value = cramers_v(contingency_table.values)
                cramer_matrix[i, j] = cramers_value

    new_corr_df = pd.DataFrame(cramer_matrix, index=columnas_categoricas, columns=columnas_categoricas)
    new_corr_df = new_corr_df.abs()

    # Obtener los valores en una tabla ordenada de mayor a menor
    correlation_values = new_corr_df.stack().reset_index()
    correlation_values.columns = ['Variable1', 'Variable2', 'Cramers_V']
    correlation_values = correlation_values.sort_values(by='Cramers_V', ascending=False)

    return correlation_values

def calculate_cramer_matrix(df, columnas_categoricas):
    cramer_matrix = np.zeros((len(columnas_categoricas), len(columnas_categoricas)))

    for i, col1 in enumerate(columnas_categoricas):
        for j, col2 in enumerate(columnas_categoricas):
            if i != j:
                contingency_table = pd.crosstab(df[col1], df[col2])
                cramers_value = cramers_v(contingency_table.values)
                cramer_matrix[i, j] = cramers_value

    new_corr_df = pd.DataFrame(cramer_matrix, index=columnas_categoricas, columns=columnas_categoricas)
    new_corr_df = new_corr_df.abs()
    new_corr_df.loc[:, :] = np.tril(new_corr_df, k=-1)
    new_corr_df = new_corr_df.stack().to_frame('correlation').reset_index().sort_values(by='correlation', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(new_corr_df.pivot(index='level_0', columns='level_1', values='correlation'), 
                annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True)
    plt.title("Matriz de Correlación (Cramers V)")
    plt.show()

def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

#NÚMRO DE NULOS Y EL PORCENTAJE
def get_percent_null_values_target(pd_loan, list_var_continuous, target):

    pd_final = pd.DataFrame()
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum()>0:
            pd_concat_percent = pd.DataFrame(pd_loan[target][pd_loan[i].isnull()]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop('index',axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_null_values'] = pd_loan[i].isnull().sum()
            pd_concat_percent['porcentaje_sum_null_values'] = pd_loan[i].isnull().sum()/pd_loan.shape[0]
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final  