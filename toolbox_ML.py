import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2_contingency, pearsonr, f_oneway

def describe_df(df):
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

# Simple funcion para clasificar y que el código quede mas bonito
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

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
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


def plot_features_num_regression(df, target_col="", umbral_corr=0, pvalue=None):
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


def get_features_cat_regression(df, target_col, pvalue=0.05):
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


def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
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