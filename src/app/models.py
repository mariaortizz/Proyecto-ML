'''Modelo para hacer las predicciones'''
import os
import pickle
import zipfile
from pathlib import Path
import traceback
import inspect
ruta_modelos = Path(os.getcwd(), 'src', 'model', 'modelos_cluster_012.zip')
ruta_cluster = Path(os.getcwd(), 'src', 'model', 'modelo_kmeans.pkl')

def extraccion_modelos(ruta_modelos):
    '''Función para extraer los modelos que tenemos cargados en un zip'''
    try:
        modelos = {}
        with zipfile.ZipFile(ruta_modelos, 'r') as zip_ref:
            for i,nombre_archivo in enumerate(zip_ref.namelist()):
                with zip_ref.open(nombre_archivo) as archivo_pickle:
                    model = pickle.load(archivo_pickle)
                    modelos[i] = model
        return modelos 
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

def extraccion_cluster(ruta_cluster):
    '''Función para extraer el cluster que tenemos en pickle'''
    try:
        with open(ruta_cluster, 'rb') as archivo:
            km = pickle.load(archivo)
        return km 
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")