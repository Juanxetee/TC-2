import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pearsonr

def describe_df(df):
    """
    Genera un resumen descriptivo del DataFrame.

    Argumentos:
    df (pandas.DataFrame): El DataFrame del que se va a generar el resumen.

    Retorna:
    pandas.DataFrame: Un DataFrame que contiene el resumen descriptivo del DataFrame de entrada.
    """
    desc = df.describe(include='all').transpose()
    desc['Missing %'] = 100 - (desc['count'] / len(df) * 100)
    desc['Cardinality %'] = (df.nunique() / len(df) * 100)
    # Agregar información sobre el tipo de datos
    desc['Data Type'] = df.dtypes
    # Reordenar las columnas para una mejor visualización
    desc = desc[['Data Type', 'Missing %', 'unique', 'Cardinality %']].rename(columns={'unique': 'Unique Values'})
    # Transponer el DataFrame
    desc = desc.transpose()
    return desc

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Sugiere el tipo de variable para cada columna del DataFrame.

    Argumentos:
    df (pandas.DataFrame): El DataFrame que se va a tipificar.
    umbral_categoria (int): Umbral para considerar una columna como categórica.
    umbral_continua (float): Umbral de cardinalidad para considerar una columna numérica como continua.

    Retorna:
    pandas.DataFrame: Un DataFrame que contiene el nombre de la columna y el tipo sugerido para cada una.
    """
    variable_types = df.nunique().apply(lambda x: 'Binaria' if x == 2 else ('Categórica' if x < umbral_categoria else ('Numérica Continua' if (x / len(df) * 100) >= umbral_continua else 'Numérica Discreta')))
    return pd.DataFrame({'nombre_variable': df.columns, 'tipo_sugerido': variable_types})

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Obtiene las características numéricas significativas para un modelo de regresión.

    Argumentos:
    df (pandas.DataFrame): El DataFrame del que se obtienen las características.
    target_col (str): El nombre de la columna que debe ser el objetivo del modelo de regresión.
    umbral_corr (float): Umbral de correlación para considerar una característica como significativa.
    pvalue (float, opcional): Valor de p para realizar los tests de hipótesis. Por defecto es None.

    Retorna:
    list or None: Una lista con las características numéricas significativas para el modelo de regresión, o None si hay errores.
    """
    if df.empty or target_col not in df.columns or not np.issubdtype(df[target_col].dtype, np.number) or not 0 <= umbral_corr <= 1:
        return None
    correlations = df.corr()[target_col].drop(target_col)
    selected_features = correlations[abs(correlations) > umbral_corr].index.tolist()
    if pvalue is not None:
        selected_features = [feature for feature in selected_features if pearsonr(df[feature], df[target_col])[1] < pvalue]
    return selected_features

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Visualiza gráficos de dispersión para características numéricas en relación con una columna objetivo de un modelo de regresión.

    Argumentos:
    df (pandas.DataFrame): El DataFrame del que se obtienen las características.
    target_col (str, opcional): El nombre de la columna que debe ser el objetivo del modelo de regresión. Por defecto es "".
    columns (list, opcional): Lista de columnas a considerar en los gráficos de dispersión. Por defecto es la lista vacía.
    umbral_corr (float, opcional): Umbral de correlación para considerar una característica como significativa. Por defecto es 0.
    pvalue (float, opcional): Valor de p para realizar el test de hipótesis. Por defecto es None.

    Retorna:
    list or None: Una lista con las características que cumplen con los criterios especificados, o None si hay errores.
    """
    selected_features = get_features_num_regression(df, target_col, umbral_corr, pvalue)
    if selected_features is None or not selected_features:
        return None
    sns.pairplot(df, vars=[target_col] + selected_features)
    plt.show()
    return selected_features

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
    if df.empty or target_col not in df.columns or not np.issubdtype(df[target_col].dtype, np.number):
        return None
    significant_features = []
    for feature in df.select_dtypes(include=['object']).columns:
        _, p_val, _, _ = chi2_contingency(pd.crosstab(df[feature], df[target_col]))
        if p_val < pvalue:
            significant_features.append(feature)
    return significant_features if significant_features else None

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
        if with_individual_plot:
            sns.histplot(data=df, x=feature, hue=target_col, multiple='stack')
            plt.title(f'Histograma agrupado para {feature} en relación con {target_col}')
            plt.xlabel(feature)
            plt.ylabel('Frecuencia')
            plt.legend(title=target_col)
            plt.show()
    return significant_features

# def card_tipo(df,umbral_categoria = 10, umbral_continua = 30):
#     # Primera parte: Preparo el dataset con cardinalidades, % variación cardinalidad, y tipos
#     df_temp = pd.DataFrame([df.nunique(), df.nunique()/len(df) * 100, df.dtypes]) # Cardinaliad y porcentaje de variación de cardinalidad
#     df_temp = df_temp.T # Como nos da los valores de las columnas en columnas, y quiero que estas sean filas, la traspongo
#     df_temp = df_temp.rename(columns = {0: "Card", 1: "%_Card", 2: "Tipo"}) # Cambio el nombre de la transposición anterior para que tengan más sentido, y uso asignación en vez de inplace = True (esto es arbitrario para el tamaño de este dataset)

#     # Corrección para cuando solo tengo un valor
#     df_temp.loc[df_temp.Card == 1, "%_Card"] = 0.00
#     # Creo la columna de sugerenica de tipo de variable, empiezo considerando todas categóricas pero podría haber empezado por cualquiera, siempre que adapte los filtros siguientes de forma correspondiente
#     df_temp["tipo_sugerido"] = "Categorica"
#     df_temp.loc[df_temp["Card"] == 2, "tipo_sugerido"] = "Binaria"
#     df_temp.loc[df_temp["Card"] >= umbral_categoria, "tipo_sugerido"] = "Numerica discreta"
#     df_temp.loc[df_temp["%_Card"] >= umbral_continua, "tipo_sugerido"] = "Numerica continua"
#     # Ojo los filtros aplicados cumplen con el enunciado pero no siguen su orden y planteamiento

#     return df_temp