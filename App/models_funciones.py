import pickle
import zipfile
import traceback
import inspect
import numpy as np
import pandas as pd


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

def extraccion_escalado(ruta_escaler):
    '''Función para extraer el cluster que tenemos en pickle'''
    try:
        with open(ruta_escaler, 'rb') as archivo:
            km = pickle.load(archivo)
        return km 
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

def aplicar_escalado(nuevo_dato1, nuevo_dato2, escaler):
    '''Funcion para escalar los datos que entran nuevos'''
    try:
        nuevos_datos = np.array([nuevo_dato1, nuevo_dato2]).reshape(1, -1)
        datos_e = escaler.transform(nuevos_datos)
        return datos_e[0][0], datos_e[0][1]
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

def transf_tipo_inmueble(x):
    '''Función para transformar los datos que recibimos'''
    try:
        valores_tipo_inmueble = {
            "piso": 0,
            "casa": 0,
            "duplex": 0,
            "atico": 0
        }

        valores_tipo_inmueble[x] = 1

        piso = valores_tipo_inmueble['piso']
        casa = valores_tipo_inmueble['casa']
        duplex = valores_tipo_inmueble['duplex']
        atico = valores_tipo_inmueble['atico']

        return piso, casa, duplex, atico
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")

def aplicar_modelo(
        CLUSTER, MODELOS, balcon, armarios, estacionamiento, trastero, 
        terraza, construccion_e,metros_e, bannos, lat, long, cee, zona, 
        piso, casa, duplex, atico
        ):
    '''Función para aplicar el cluster a los datos nuevos que tenemos'''
    try:
        cluster_pred = CLUSTER.predict([[
            balcon, armarios, estacionamiento, trastero, 
            terraza, construccion_e,metros_e, bannos, 
            lat, long, cee, zona, piso, casa, duplex, atico
            ]])

        data = {'balcon' : [balcon],
                            'armarios_empotrados' : [armarios], 
                            'estacionamiento' : [estacionamiento], 
                            'trastero' : [trastero],
                            'terraza' : [terraza], 
                            'annio_construccion' : [construccion_e], 
                            'metros_cuadrados' : [metros_e],
                            'bannos' : [bannos],
                            'latitud' : [lat],
                            'longitud' : [long],
                            'cee_e' : [cee],
                            'zona_e' : [zona],
                            'tipo_inmueble_HouseType 1: Pisos' : [piso],
                            'tipo_inmueble_HouseType 2: Casa o chalet' : [casa],
                            'tipo_inmueble_HouseType 4: Dúplex' : [duplex],
                            'tipo_inmueble_HouseType 5: Áticos' : [atico],
                            'Cluster' : [cluster_pred[0]]
        }
        valores = pd.DataFrame(data)

        model = MODELOS[cluster_pred[0]] 
        print(model)
        print(valores)
        precio = model.predict(valores)
        print(precio)
        return precio, cluster_pred[0]
    except Exception as a:
        traceback.print_exc()
        func = inspect.stack()[1].function
        print(f"No se pudo terminar el proceso en la función {func} por {a}")




