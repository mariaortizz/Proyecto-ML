'''Archivo por el que corre la aplicación'''
from flask import Flask, render_template, request
import os 
from models import extraccion_modelos, extraccion_cluster
from pathlib import Path

ruta_modelos = Path(os.getcwd(), 'src', 'model', 'modelos_cluster_012.zip')
ruta_cluster = Path(os.getcwd(), 'src', 'model', 'modelo_kmeans.pkl')

MODELOS = extraccion_modelos(ruta_modelos)
CLUSTER = extraccion_cluster(ruta_cluster)

'''
'balcon' : 1/0, 
'armarios_empotrados' : 1/0, 
'estacionamiento' : 1/0, 
'trastero' : 1/0,
'terraza' : 1/0, 
'annio_construccion' : valor, 
'metros_cuadrados' : valor, 
'bannos' : valor, 
'latitud' : valor, 
'longitud' : valor, 
'cee_e' : valor, 
'zona_e' : valor,
'tipo_inmueble_HouseType 1: Pisos' : 1/0,
'tipo_inmueble_HouseType 2: Casa o chalet' : 1/0,
'tipo_inmueble_HouseType 4: Dúplex' : 1/0,
'tipo_inmueble_HouseType 5: Áticos' : 1/0
'''

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__, instance_relative_config=True)
app.config.from_object("config")

@app.route('/', methods = ['GET'])
def home():
    return render_template("home.html")

# @app.route('/predict', methods = ['POST'])
# def predict(balcon, amrarios, estacionamiento, trastero, terraza, construccion, metros, 
#             bannos, lat, long, cee, zona, piso, casa, duplex, atico):
#     balcon = float(request.args.get('a', None))
#     amrarios
#     estacionamiento
#     trastero
#     terraza
#     construccion
#     metros
#     bannos
#     lat
#     long
#     cee
#     zona
#     piso
#     casa
#     duplex
#     atico
#     a = float(request.args.get('a', None))
#     b = float(request.args.get('b', None))
#     c = float(request.args.get('c', None))
#     d = float(request.args.get('d', None))
#     cluster = CLUSTER.predict([[balcon, amrarios, estacionamiento, trastero, terraza, 
#                              construccion, metros, bannos, lat, long, cee, zona, 
#                              piso, casa, duplex, atico]])
#     prediccion = m.predict([[balcon, amrarios, estacionamiento, trastero, terraza, 
#                              construccion, metros, bannos, lat, long, cee, zona, 
#                              piso, casa, duplex, atico]])
#     print(app.config['VARIABLE'])
#     return render_template("predict.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)