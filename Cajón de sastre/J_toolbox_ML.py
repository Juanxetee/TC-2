"""
toolbox_ML.py

Este módulo contiene funciones útiles para preparar y analizar datos para modelos de Machine Learning.
Incluye funciones para describir datos, tipificar variables, calcular correlaciones y generar visualizaciones,
además de identificar características significativas tanto numéricas como categóricas.

Dependencias:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def describe_df(df):
    """ Función para describir un DataFrame. """
    resumen = pd.DataFrame({
        'Tipo de dato': df.dtypes,
        'Porcentaje de nulos': df.isnull().mean() * 100,
        'Valores únicos': df.nunique()
    })
    resumen['Porcentaje de cardinalidad'] = (resumen['Valores únicos'] / len(df)) * 100
    return resumen.transpose()

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """ Función para clasificar tipos de variables basado en la cardinalidad y el porcentaje de cardinalidad. """
    resumen = pd.DataFrame({
        'nombre_variable': df.columns,
        'cardinalidad': df.nunique(),
        'porcentaje_cardinalidad': (df.nunique() / len(df)) * 100
    })
    condiciones = [
        (resumen['cardinalidad'] == 2),
        (resumen['cardinalidad'] < umbral_categoria),
        (resumen['cardinalidad'] >= umbral_categoria) & (resumen['porcentaje_cardinalidad'] >= umbral_continua),
        (resumen['cardinalidad'] >= umbral_categoria) & (resumen['porcentaje_cardinalidad'] < umbral_continua)
    ]
    tipos = ['Binaria', 'Categórica', 'Numérica Continua', 'Numérica Discreta']
    resumen['tipo_sugerido'] = pd.np.select(condiciones, tipos)
    return resumen[['nombre_variable', 'tipo_sugerido']]

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """ Función para obtener características numéricas significativas respecto a una columna objetivo. """
    if not 0 <= umbral_corr <= 1:
        print("Error: El umbral de correlación debe estar entre 0 y 1.")
        return None
    if pvalue is not None and not 0 < pvalue < 1:
        print("Error: pvalue debe estar entre 0 y 1, exclusivo.")
        return None
    features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != target_col:
            corr, pval = stats.pearsonr(df[col].dropna(), df[target_col].dropna())
            if abs(corr) >= umbral_corr and (pvalue is None or pval <= pvalue):
                features.append(col)
    return features

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """ Función para plotear pares de características numéricas que tienen una correlación significativa. """
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    valid_columns = []
    for col in columns:
        if col != target_col:
            corr, pval = stats.pearsonr(df[col].dropna(), df[target_col].dropna())
            if abs(corr) >= umbral_corr and (pvalue is None or pval <= pvalue):
                valid_columns.append(col)
    if valid_columns:
        valid_columns.append(target_col)
        num_plots = len(valid_columns) // 5 + (1 if len(valid_columns) % 5 else 0)
        for i in range(num_plots):
            sns.pairplot(df, vars=valid_columns[i*5:(i+1)*5])
            plt.show()
    return valid_columns

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """ Función para identificar variables categóricas significativamente relacionadas con una columna objetivo numérica. """
    if not 0 < pvalue < 1:
        print("Error: pvalue debe estar entre 0 y 1, exclusivo.")
        return None
    significant_cats = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        grouped = df.groupby(col)[target_col].apply(list)
        f_val, p_val = stats.f_oneway(*grouped)
        if p_val <= pvalue:
            significant_cats.append(col)
    return significant_cats

def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Genera histogramas agrupados de la variable objetivo para cada valor de las variables categóricas
    incluidas en 'columns' que tienen una relación estadísticamente significativa con la variable objetivo.

    Argumentos:
    df (DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la columna objetivo numérica en el DataFrame.
    columns (list): Lista de nombres de columnas categóricas a considerar para los histogramas.
    pvalue (float): Umbral de p-value para considerar la relación estadísticamente significativa.
    with_individual_plot (bool): Si es True, genera un plot por cada variable categórica válida.

    Retorna:
    list: Lista de nombres de columnas categóricas que cumplen con el criterio de significancia.
    """
    # Verificaciones de entrada
    if not 0 < pvalue < 1:
        print("Error: pvalue debe estar entre 0 y 1, exclusivo.")
        return None
    if target_col not in df.columns or df[target_col].dtype not in [np.float64, np.int64]:
        print("Error: target_col debe ser numérica y existir en el DataFrame.")
        return None

    if not columns:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    significant_cats = []
    for col in columns:
        if df[col].dtype.name in ['object', 'category']:
            # Realizar ANOVA entre la columna categórica y la variable objetivo
            grouped = df.groupby(col)[target_col].apply(list)
            f_val, p_val = stats.f_oneway(*grouped)
            if p_val <= pvalue:
                significant_cats.append(col)
                if with_individual_plot:
                    # Generar histogramas para cada categoría
                    sns.catplot(x=col, y=target_col, kind="bar", data=df)
                    plt.title(f'Relación entre {col} y {target_col}')
                    plt.show()

    return significant_cats


