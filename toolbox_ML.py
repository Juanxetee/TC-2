# IMPORTS

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
from math import ceil

# Preprocesado y modelado
# ==============================================================================
from sklearn.model_selection import train_test_split
from scipy.stats import f_oneway
import scipy.stats as stats
from scipy.stats import chi2_contingency, pearsonr, f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, classification_report, 
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE, mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------

################################################################################################
##################################### TOOLBOX 1 ################################################
################################################################################################

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Describe el dtype de cada columna, los valores nulos en %, quantos valores únicos hay en la columna y el % de cardinalidad.

    Argumentos:
    data(pd.DataFrame): DataFrame de Pandas inicial

    Retorna:
    pd.DataFrame: Data inicial transformado con los valores descritos   
    '''
    resumen = pd.DataFrame({
        'Tipo de dato': df.dtypes,
        'Porcentaje de nulos': round(df.isnull().mean() * 100,2),
        'Valores únicos': df.nunique()
    })
    resumen['Porcentaje de cardinalidad'] = round(resumen['Valores únicos'] / len(df) * 100,2)
    return resumen.transpose()

# ------------------------------------------------------------------------------

# Simple funcion para clasificar y que el código de tipifica_variable quede mas bonito
def _classify(data: pd.DataFrame, key: str,  umbral_categoria:int, umbral_continua:float) -> str: 
    cardi = data[key].nunique() 
    if cardi == 2: # ¿La cardinalidad es igual que dos?
        return "Binaria"
    elif cardi < umbral_categoria: # ¿La cardinalidad es mas pequeña que el número que escogemos para considerar una variable categórica?
        return "Categórica"
    elif cardi/len(data[key])*100 >= umbral_continua: # ¿El % de la cardinalidad es mayor o igual que el número que escogemos para delimitar cuando és Continua o Discreta?
        return "Numérica Continua"
    else:
        return "Numérica Discreta"
      
def tipifica_variable (data:pd.DataFrame, umbral_categoria:int, umbral_continua:float) -> pd.DataFrame:
    '''
    Tipo de variable de cada columna según su cardinalidad.

    Argumentos:
    data(pd.DataFrame): DataFrame inicial
    umbral_categoria(int): Número que escogemos para delimitar a partir de cuanto consideramos que es una variable categorica
    umbral_continua(float): Número que escogemos para delimitar a partir de cuanto una variable numérica es discreta
    
    Retorna:
    pd.DataFrame: Data inicial transformado   
    '''
    # Diccionario con los resultados de las preguntas sobre la cardinalidad
    dic_tip_var = {
        "tipo_sugerido": [_classify(data, key, umbral_categoria, umbral_continua) for key in data]
    }
    # Añadimos un extra, simple print para tener en cuenta si hay valores nulos no tratados en el dataframe
    for x in data:
        hay_nulos = data[x].isnull().sum()
        if hay_nulos != 0:
            print(f'OJO! En la columna "{x}" hay valores nulos.')

    return pd.DataFrame(dic_tip_var, index=[x for x in data])

# ------------------------------------------------------------------------------

def get_features_num_regression(df:pd.DataFrame, target_col:str, umbral_corr:float, pvalue:float=None)-> list: 
    """
    Esta función devuelve las columnas numéricas del dataframe que tienen una correlación significativa con la columna de target.

    Argumentos:
    df (DataFrame): El dataframe a analizar.
    target_col (str): Nombre de la columna que será el target.
    umbral_corr (float): Umbral de correlación para considerar una variable como relevante.
    pvalue (float, opcional): Valor p para el test de hipótesis. Por defecto es None.

    Retorna:
    list: Una lista con los nombres de las columnas numéricas que cumplen con los criterios de correlación y p-value.
    """

    if umbral_corr > 1 or umbral_corr < 0:
        print("Error: 'umbral_corr' debe ser un valor entre 0 y 1.")
        return None
    
    if pvalue is not None and (pvalue <= 0 or pvalue >= 1):
        print("Error: 'pvalue' debe ser un valor entre 0 y 1.")
        return None
        
    features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != target_col:
            corr, pval = stats.pearsonr(df[col].dropna(), df[target_col].dropna())
            if abs(corr) >= umbral_corr and (pvalue is None or pval <= pvalue):
                features.append(col)
    return features

# ------------------------------------------------------------------------------

def plot_features_num_regression(df:pd.DataFrame, target_col:str="", umbral_corr:float=0, pvalue:float=None):
    """
    Visualiza gráficos de dispersión para características numéricas en relación con una columna objetivo de un modelo de regresión.

    Argumentos:
    df (pandas.DataFrame): El DataFrame del que se obtienen las características.
    target_col (str, opcional): El nombre de la columna que debe ser el objetivo del modelo de regresión. Por defecto es "".
    umbral_corr (float, opcional): Umbral de correlación para considerar una característica como significativa. Por defecto es 0.
    pvalue (float, opcional): Valor de p para realizar el test de hipótesis. Por defecto es None.

    Retorna:
    list or None: Una lista con las características que cumplen con los criterios especificados, o None si hay errores.
    """
    selected_features = get_features_num_regression(df, target_col, umbral_corr, pvalue)
    fig = sns.pairplot(df, vars= selected_features, hue = target_col)
    plt.show()
    return fig

# ------------------------------------------------------------------------------

def get_features_cat_regression(df:pd.DataFrame, target_col:str, pvalue:float=0.05) -> list:
    """
    Obtiene las características categóricas significativas para un modelo de regresión.

    Argumentos:
    df (pandas.DataFrame): El DataFrame del que se obtienen las características.
    target_col (str): El nombre de la columna que debe ser el objetivo del modelo de regresión.
    pvalue (float, opcional): Valor de p para realizar el test de hipótesis. Por defecto es 0.05.

    Retorna:
    list or None: Una lista con las características categóricas significativas para el modelo de regresión, o None si hay errores.
    """
    if pvalue <= 0 or pvalue >= 1:
        print("Error: 'pvalue' debe ser un valor entre 0 y 1.")
        return None
    
    significant_cats = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        grouped = df.groupby(col, observed=True)[target_col].apply(list)
        f_val, p_val = stats.f_oneway(*grouped)
        if p_val <= pvalue:
            significant_cats.append(col)
    return significant_cats

# ------------------------------------------------------------------------------

def plot_features_cat_regression(df:pd.DataFrame, target_col:str="", columns:list=[], pvalue:float=0.05, with_individual_plot:bool=False):
    """
    Visualiza histogramas agrupados para características categóricas en relación con una columna objetivo de un modelo de regresión.

    Argumentos:
    df (pandas.DataFrame): El DataFrame del que se obtienen las características.
    target_col (str, opcional): El nombre de la columna que debe ser el objetivo del modelo de regresión. Por defecto es "".
    columns (list, opcional): Lista de columnas a considerar en los histogramas agrupados. Por defecto es la lista vacía.
    pvalue (float, opcional): Valor de p para realizar el test de hipótesis. Por defecto es 0.05.
    with_individual_plot (bool, opcional): Indica si se deben mostrar los histogramas individuales de cada variable categórica. Por defecto es False.

    Retorna:
    list or None: Una lista con las características que cumplen con los criterios especificados, o None si hay errores.
    """
    significant_features = get_features_cat_regression(df, target_col, pvalue)
    if significant_features is None or not significant_features:
        return None
    for feature in significant_features:
        #if with_individual_plot:
        fig = sns.histplot(data=df, x=feature, hue=target_col, multiple='stack')
        plt.title(f'Histograma agrupado para {feature} en relación con {target_col}')
        plt.xlabel(feature)
        plt.ylabel('Frecuencia')
        plt.show()
    return fig

################################################################################################
##################################### TOOLBOX 2 ################################################
################################################################################################

def eval_model(target:pd.Series, predictions:pd.Series, problem_type:str, metrics:list)-> tuple:
    """
    Evalúa un modelo de machine learning en función de las métricas especificadas.

    Argumentos:
    target (array-like): Valores reales del target.
    predictions (array-like): Valores predichos por el modelo.
    problem_type (str): Tipo de problema ('regression' o 'classification').
    metrics (list): Lista de métricas a calcular (RMSE, MAE, MAPE, GRAPH, ACCURACY, PRECISION, RECALL, CLASS_REPORT, MATRIX, MATRIX_RECALL, MATRIX_PRED, PRECISION_X, RECALL_X).

    Retorna:
    tuple: Tupla de métricas calculadas en el orden especificado.
    """
    results = []
    
    if problem_type == 'regression':
        for metric in metrics:
            if metric == 'RMSE':
                rmse = np.sqrt(mean_squared_error(target, predictions))
                print(f"RMSE: {rmse}")
                results.append(rmse)
            elif metric == 'MAE':
                mae = mean_absolute_error(target, predictions)
                print(f"MAE: {mae}")
                results.append(mae)
            elif metric == 'MAPE':
                try:
                    mape = mean_absolute_percentage_error(target, predictions)
                    print(f"MAPE: {mape}")
                    results.append(mape)
                except ValueError as e:
                    raise ValueError("MAPE no se puede calcular, asegúrese de que los valores del target no sean cero.") from e
            elif metric == 'GRAPH':
                plt.scatter(target, predictions)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title("Actual vs Predicted")
                plt.show()
                
    elif problem_type == 'classification':
        for metric in metrics:
            if metric == 'ACCURACY':
                accuracy = accuracy_score(target, predictions)
                print(f"Accuracy: {accuracy}")
                results.append(accuracy)
            elif metric == 'PRECISION':
                precision = precision_score(target, predictions, average='macro')
                print(f"Precision: {precision}")
                results.append(precision)
            elif metric == 'RECALL':
                recall = recall_score(target, predictions, average='macro')
                print(f"Recall: {recall}")
                results.append(recall)
            elif metric == 'CLASS_REPORT':
                report = classification_report(target, predictions)
                print("Classification Report:\n", report)
            elif metric == 'MATRIX':
                cm = confusion_matrix(target, predictions)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.show()
            elif metric == 'MATRIX_RECALL':
                cm = confusion_matrix(target, predictions, normalize='true')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.show()
            elif metric == 'MATRIX_PRED':
                cm = confusion_matrix(target, predictions, normalize='pred')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.show()
            elif metric.startswith('PRECISION_'):
                class_label = metric.split('_')[1]
                precision_class = precision_score(target, predictions, labels=[class_label], average='macro', zero_division=0)
                if precision_class == 0:
                    raise ValueError(f"Etiqueta '{class_label}' no existe en el target.")
                print(f"Precision for {class_label}: {precision_class}")
                results.append(precision_class)
            elif metric.startswith('RECALL_'):
                class_label = metric.split('_')[1]
                recall_class = recall_score(target, predictions, labels=[class_label], average='macro', zero_division=0)
                if recall_class == 0:
                    raise ValueError(f"Etiqueta '{class_label}' no existe en el target.")
                print(f"Recall for {class_label}: {recall_class}")
                results.append(recall_class)
    
    return tuple(results)

# ------------------------------------------------------------------------------

def get_features_num_classification(df:pd.DataFrame, target_col:str, pvalue:float=0.05) -> list:
    """
    Selecciona columnas numéricas que tienen una relación significativa con la columna target
    utilizando ANOVA para problemas de clasificación.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre de la columna objetivo (target) que debe ser categórica.
    pvalue (float): Valor de significancia para el test de ANOVA (por defecto 0.05).

    Retorna:
    list: Lista de columnas numéricas que pasan el test de ANOVA.
    """
    # Comprobación de que target_col existe en el dataframe
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no existe en el DataFrame.")
        return None
    
    # Comprobación de que target_col es categórica
    if not pd.api.types.is_categorical_dtype(df[target_col]) and not pd.api.types.is_object_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es categórica.")
        return None
    
    # Obtener columnas numéricas
    numeric_cols = df.select_dtypes(include='number').columns
    
    # Lista para almacenar las columnas que pasan el test de ANOVA
    significant_features = []
    
    # Realizar ANOVA para cada columna numérica
    for col in numeric_cols:
        groups = [df[col][df[target_col] == category].dropna() for category in df[target_col].unique()]
        
        # Verificar si todas las listas en groups son constantes
        if any(len(group) == 0 or group.nunique() <= 1 for group in groups):
            print(f"La columna '{col}' tiene grupos con valores constantes o vacíos y no se puede aplicar ANOVA.")
            continue
        
        try:
            # Realizar ANOVA
            f_val, p_val = f_oneway(*groups)
        except Exception as e:
            print(f"No se pudo realizar ANOVA para la columna '{col}': {e}")
            continue
        
        # Verificar si el p_valor es menor que 1 - pvalue
        if p_val < pvalue:
            significant_features.append(col)
    
    return significant_features
# ------------------------------------------------------------------------------

def plot_features_num_classification(df:pd.DataFrame, target_col:str="", columns:list=[], pvalue:float=0.05) -> list:
    """
    Genera pairplots de características numéricas significativas con respecto a una columna objetivo categórica.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre de la columna objetivo (target) que debe ser categórica o numérica discreta.
    columns (list): Lista de nombres de columnas numéricas a considerar (por defecto lista vacía).
    pvalue (float): Valor de significancia para el test de ANOVA (por defecto 0.05).

    Retorna:
    list: Lista de columnas que pasan el test de ANOVA.
    """
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_categorical_dtype(df[target_col]) and not pd.api.types.is_integer_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es categórica ni numérica discreta.")
        return None

    if not columns:
        columns = df.select_dtypes(include='number').columns.tolist()

    significant_features = get_features_num_classification(df, target_col, pvalue)
    if not significant_features:
        print("No hay columnas significativas para el nivel de significancia proporcionado.")
        return []

    # Añadir la columna objetivo a las columnas significativas para el pairplot
    significant_features.append(target_col)

    # Crear pairplot
    sns.pairplot(df[significant_features], hue=target_col)
    plt.show()

    return significant_features

# ------------------------------------------------------------------------------

def get_features_cat_classification(df:pd.DataFrame, target_col:str, normalize:bool=False, mi_threshold:float=0) -> list:
    """
    Selecciona columnas categóricas que tienen una relación significativa con la columna target
    utilizando Mutual Information para problemas de clasificación.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre de la columna objetivo (target) que debe ser categórica o numérica discreta.
    normalize (bool): Indica si se debe normalizar la información mutua (por defecto False).
    mi_threshold (float): Umbral de información mutua para seleccionar las características (por defecto 0).

    Retorna:
    list: Lista de columnas categóricas que cumplen con el umbral de información mutua.
    """
    # Comprobación de que target_col existe en el dataframe
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobación de que target_col es categórica o numérica discreta
    if not isinstance(df[target_col].dtype, pd.CategoricalDtype) and not pd.api.types.is_integer_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es categórica ni numérica discreta.")
        return None

    # Comprobación de que mi_threshold es un float y en el rango correcto si normalize es True
    if normalize and (not isinstance(mi_threshold, float) or mi_threshold < 0 or mi_threshold > 1):
        print(f"El umbral de información mutua '{mi_threshold}' debe ser un float entre 0 y 1 cuando normalize es True.")
        return None

    # Obtener columnas categóricas
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    
    # Convertir columnas categóricas a códigos numéricos
    df_encoded = df[categorical_cols].apply(lambda x: x.astype('category').cat.codes)

    # Calcular Mutual Information
    mi = mutual_info_classif(df_encoded, df[target_col], discrete_features=True)

    # Normalizar si es necesario
    if normalize:
        total_mi = mi.sum()
        mi = mi / total_mi

    # Seleccionar columnas que cumplen con el umbral de información mutua
    selected_features = [col for col, value in zip(categorical_cols, mi) if value >= mi_threshold]

    return selected_features

# ------------------------------------------------------------------------------

def plot_features_cat_classification(df:pd.DataFrame, target_col:str="", columns:list=[], mi_threshold:float=0.0, normalize:bool=False):
    """
    Genera subplots de características categóricas significativas con respecto a una columna objetivo categórica.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre de la columna objetivo (target) que debe ser categórica o numérica discreta.
    columns (list): Lista de nombres de columnas categóricas a considerar (por defecto lista vacía).
    mi_threshold (float): Umbral de información mutua para seleccionar las características (por defecto 0.0).
    normalize (bool): Indica si se debe normalizar la información mutua (por defecto False).

    Retorna:
    list: Lista de columnas que pasan el umbral de información mutua.
    """
    # Comprobación de que target_col existe en el dataframe
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobación de que target_col es categórica o numérica discreta
    if not isinstance(df[target_col].dtype, pd.CategoricalDtype) and not pd.api.types.is_integer_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es categórica ni numérica discreta.")
        return None

    # Comprobación de que mi_threshold es un float y en el rango correcto si normalize es True
    if normalize and (not isinstance(mi_threshold, float) or mi_threshold < 0 or mi_threshold > 1):
        print(f"El umbral de información mutua '{mi_threshold}' debe ser un float entre 0 y 1 cuando normalize es True.")
        return None

    # Si la lista de columnas está vacía, obtener todas las columnas categóricas del dataframe
    if not columns:
        columns = df.select_dtypes(include=['category', 'object']).columns.tolist()

    # Filtrar columnas categóricas
    categorical_cols = [col for col in columns if col in df.columns and isinstance(df[col].dtype, pd.CategoricalDtype)]

    # Convertir columnas categóricas a códigos numéricos
    df_encoded = df[categorical_cols].apply(lambda x: x.astype('category').cat.codes)

    # Calcular Mutual Information
    mi = mutual_info_classif(df_encoded, df[target_col], discrete_features=True)

    # Normalizar si es necesario
    if normalize:
        total_mi = mi.sum()
        mi = mi / total_mi

    # Seleccionar columnas que cumplen con el umbral de información mutua
    significant_features = [col for col, value in zip(categorical_cols, mi) if value >= mi_threshold]

    if not significant_features:
        print("No hay columnas significativas para el nivel de significancia proporcionado.")
        return []

    # Crear subplots
    num_features = len(significant_features)
    num_cols = 2  # Número de columnas en los subplots
    num_rows = (num_features + 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axs = axs.flatten()

    for i, col in enumerate(significant_features):
        sns.countplot(data=df, x=col, hue=target_col, ax=axs[i])
        axs[i].set_title(f"Distribución de {col} respecto a {target_col}")

    for i in range(num_features, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

    return significant_features

# ------------------------------------------------------------------------------ EXTRA

def super_selector(dataset:pd.DataFrame, target_col:str="", selectores:dict=None, hard_voting:list=[]) -> dict:
    """
    Selecciona características de un DataFrame utilizando diferentes métodos de selección y aplica votación dura.

    Argumentos:
    dataset (pd.DataFrame): DataFrame con las características.
    target_col (str): Nombre de la columna objetivo (target) que puede ser numérica o categórica (por defecto "").
    selectores (dict): Diccionario con los métodos de selección y sus parámetros (por defecto None).
    hard_voting (list): Lista con las características para considerar en la votación dura (por defecto lista vacía).

    Retorna:
    dict: Diccionario con las listas de características seleccionadas por cada método y la votación dura.
    """
    if target_col == "" or target_col not in dataset.columns:
        print(f"La columna target '{target_col}' no es válida o no está en el DataFrame.")
        return None

    if selectores is None or not selectores:
        non_target_cols = [col for col in dataset.columns if col != target_col and dataset[col].nunique() != 1 and dataset[col].nunique() != len(dataset)]
        return {"all_features": non_target_cols}

    result = {}

    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]

    # Convertir todas las variables categóricas a códigos numéricos
    X = X.apply(lambda x: x.astype('category').cat.codes if x.dtype == 'object' or isinstance(x.dtype, pd.CategoricalDtype) else x)

    # Escalar las características numéricas
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])

    # KBest
    if "KBest" in selectores:
        k = selectores["KBest"]
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        kbest_features = list(X.columns[selector.get_support()])
        result["KBest"] = kbest_features

    # FromModel
    if "FromModel" in selectores:
        model, threshold = selectores["FromModel"]
        if isinstance(threshold, int):
            selector = SelectFromModel(estimator=model, max_features=threshold, threshold=-np.inf)
        else:
            selector = SelectFromModel(estimator=model, threshold=threshold)
        selector.fit(X, y)
        from_model_features = list(X.columns[selector.get_support()])
        result["FromModel"] = from_model_features

    # RFE
    if "RFE" in selectores:
        model, n_features, step = selectores["RFE"]
        model.set_params(max_iter=1000)  # Aumentar el número de iteraciones
        selector = RFE(estimator=model, n_features_to_select=n_features, step=step)
        selector.fit(X, y)
        rfe_features = list(X.columns[selector.get_support()])
        result["RFE"] = rfe_features

    # SFS
    if "SFS" in selectores:
        model, k_features = selectores["SFS"]
        selector = SFS(estimator=model, k_features=k_features, forward=True, floating=False, scoring='accuracy', cv=5)
        selector.fit(X, y)
        sfs_features = list(selector.k_feature_names_)
        result["SFS"] = sfs_features

    all_features = []
    if "KBest" in result:
        all_features.extend(result["KBest"])
    if "FromModel" in result:
        all_features.extend(result["FromModel"])
    if "RFE" in result:
        all_features.extend(result["RFE"])
    if "SFS" in result:
        all_features.extend(result["SFS"])
    all_features.extend(hard_voting)

    feature_counts = pd.Series(all_features).value_counts()
    top_features = feature_counts.index.tolist()

    result["hard_voting"] = top_features

    return result