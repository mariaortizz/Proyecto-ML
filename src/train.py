
#librerias estandar
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import traceback
import inspect
import time

#librerias modelos
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import  RandomForestRegressor,  VotingRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor

#cluster
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import silhouette_visualizer
from sklearn.cluster import KMeans 

pd.set_option('display.max_columns', None)
# sys.path.append(r'PRIVADO_MARIA\Proyecto_ML\src')

#modulos propios
from utils import funciones as f

#rutas
datos_limpios_path = r'data\processed\df_limpio.csv' 

def timeit(func):
    def envoltorio(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} - Tiempo transcurrido: {end_time - start_time} segundos")
        return result
    return envoltorio

@timeit
def extraccion_datos(datos_limpios_path):
    '''Función para extraer los datos del CSV que limpio previamente'''
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        datos_limpios_path = os.path.join(script_dir, datos_limpios_path)
        df = pd.read_csv(datos_limpios_path, index_col='Unnamed: 0')
        return df
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def dividir_datos(df):
    '''Función para dividir los datos del df obtenido antes'''
    try:
        X = df.drop(columns=['precio_venta_por_m2'], axis=1)
        y = df['precio_venta_por_m2']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 33)

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        return X_train, y_train, X_test, y_test
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def tratar_cee(X_train):
    '''Func para tratar la columna CEE'''
    try:
        o_encoder = OrdinalEncoder(categories=[["no indicado", "inmueble exento", 
                                      "en trámite", "G", "F", "E", 
                                      "D", "C", "B", "A" 
                                      ]])

        o_encoder.fit(X_train[["cee"]])
        X_train["cee_e"] = o_encoder.transform(X_train[["cee"]])
        X_train.drop(columns=['cee'], inplace=True)
        #devolvemos el o_encoder porque después lo tenemos que usar para el test
        
        return X_train, o_encoder
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def tratar_zona(X_train):
    '''Función para tratar la columnas zona'''
    try:
        l_encoder = LabelEncoder()
        X_train['zona_e'] = l_encoder.fit_transform(X_train['zona'])
        X_train.drop(columns=['zona', 'ubicacion'], inplace=True)

        return X_train, l_encoder

    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def tratar_tipo(X_train):
    '''Función para tratar el tipo de inmueble'''
    try:
        oh_encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        encoded = oh_encoder.fit_transform(X_train[['tipo_inmueble']])
        encoded_df = pd.DataFrame(encoded, columns=oh_encoder.get_feature_names_out(['tipo_inmueble']))

        X_train = pd.concat([X_train, encoded_df], axis=1)
        X_train = X_train.drop(['tipo_inmueble'], axis=1)

        return X_train, oh_encoder
    
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def escalado(X_train):
    '''Función para escalar las columnas'''
    try:
        columnas_esc = ['annio_construccion', 'metros_cuadrados']
        scaler = RobustScaler()
        X_train[columnas_esc] = scaler.fit_transform(X_train[columnas_esc])

        return X_train, scaler
    
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def procesamiento_train(X_train):
    '''Función para tratar los datos de train'''
    try:
        X_train_0, o_encoder = tratar_cee(X_train)
        X_train_1, l_encoder = tratar_zona(X_train_0)
        X_train_2, oh_encoder = tratar_tipo(X_train_1)
        X_train_3 = f.tranformacion_numerica(X_train_2)
        X_train_4, scaler = escalado(X_train_3)

        X_train_4.drop(columns=['piso', 'precio_venta', 'jardin', 
                              'zonas_verdes', 'accesible', 'bajo',
                                'pileta', 'exterior', 'necesita_reforma', 
                                'habitaciones'], inplace=True)
        
        return X_train_4, o_encoder, l_encoder, oh_encoder, scaler
    
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def cluster(X_train, y_train):
    '''Función para aplicar un cluster'''
    try:
        km_0 = KMeans(n_clusters=3, random_state=42)
        km_0.fit(X_train)
        predd = km_0.predict(X_train)

        df_c = pd.DataFrame(predd, columns=['Cluster'])
        X_train_clust = pd.concat([X_train, df_c], axis=1)

        df_unido = pd.concat([X_train_clust, y_train], axis=1)

        df_unido_0 = df_unido[df_unido['Cluster'] == 0]
        df_unido_1 = df_unido[df_unido['Cluster'] == 1]
        df_unido_2 = df_unido[df_unido['Cluster'] == 2]

        return df_unido_0, df_unido_1, df_unido_2, km_0
    
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def cluster_0(df_unido_0):
    '''Funcion para entrenar el cluster 0'''
    try:
        X_train_0 = df_unido_0.drop(columns='precio_venta_por_m2')
        y_train_0 = df_unido_0['precio_venta_por_m2']

        model_cl0_111 = RandomForestRegressor()
        model_cl0_222 = HistGradientBoostingRegressor(learning_rate = 0.1,
                                    max_depth = 5,
                                    max_iter = 200,
                                    max_leaf_nodes = 50,
                                    min_samples_leaf = 5)
        model_cl0_333 = XGBRegressor(learning_rate = 0.1,
                            max_depth = 9,
                            min_child_weight = 5,
                            n_estimators = 100)
        base_models=[
                            ('rfr', model_cl0_111),
                            ('hs', model_cl0_222),
                            ('xgb', model_cl0_333)
                        ]
        meta_model = LinearRegression()
        model_555 = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        model_c_0 = model_555

        model_c_0.fit(X_train_0, y_train_0)

        return model_c_0
    
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def cluster_1(df_unido_1):
    '''Funcion para entrenar el cluster 1'''
    try:
        X_train_1 = df_unido_1.drop(columns='precio_venta_por_m2')
        y_train_1 = df_unido_1['precio_venta_por_m2']

        model_cl1_111 = RandomForestRegressor()
        model_cl1_222 = HistGradientBoostingRegressor(learning_rate = 0.1,
                                    max_depth = 10,
                                    max_iter = 200,
                                    max_leaf_nodes = 50,
                                    min_samples_leaf = 5)
        model_cl1_333 = XGBRegressor(learning_rate = 0.1,
                            max_depth = 6,
                            min_child_weight = 5,
                            n_estimators = 100)
        base_models=[
                            ('rfr', model_cl1_111),
                            ('hs', model_cl1_222),
                            ('xgb', model_cl1_333)
                        ]

        meta_model = LinearRegression()

        model_555 = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        model_c_1 = model_555

        model_c_1.fit(X_train_1, y_train_1)

        return model_c_1
    
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

@timeit
def cluster_2(df_unido_2):
    '''Funcion para entrenar el cluster 2'''
    try:
        X_train_2 = df_unido_2.drop(columns='precio_venta_por_m2')
        y_train_2 = df_unido_2['precio_venta_por_m2']

        model_cl2_111 = RandomForestRegressor()
        model_cl2_222 = HistGradientBoostingRegressor()
        model_cl2_333 = XGBRegressor(learning_rate = 0.1,
                            max_depth = 7,
                            min_child_weight = 5,
                            n_estimators = 100)
        base_models=[
                            ('rfr', model_cl2_111),
                            ('hs', model_cl2_222),
                            ('xgb', model_cl2_333)
                        ]

        model_444 = VotingRegressor(
                        estimators=base_models
                    )

        model_c_2 = model_444

        model_c_2.fit(X_train_2, y_train_2)

        return model_c_2
    
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

if __name__ == "__main__":
    try:
        df = extraccion_datos(datos_limpios_path)
        X_train_0, y_train, X_test, y_test = dividir_datos(df)
        X_train, o_encoder, l_encoder, oh_encoder, scaler = procesamiento_train(X_train_0)
        df_0, df_1, df_2, km_0 = cluster(X_train, y_train)
        modelo_0 = cluster_0(df_0)
        modelo_1 = cluster_1(df_1)
        modelo_2 = cluster_2(df_2)
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No puede entrenar los modelos por la {func} - {a}")

print('Modelos entrenados - proceso terminado')
time.sleep(3)