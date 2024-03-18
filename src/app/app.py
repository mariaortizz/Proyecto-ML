'''Archivo por el que corre la aplicación'''
from flask import Flask, render_template, request
import os 
from pathlib import Path
from models_funciones import extraccion_cluster, extraccion_escalado, \
    extraccion_modelos, aplicar_escalado, transf_tipo_inmueble, aplicar_modelo

ruta_modelos = Path(os.getcwd(), 'src', 'model', 'modelos_cluster_012.zip')
ruta_cluster = Path(os.getcwd(), 'src', 'model', 'modelo_kmeans.pkl')
ruta_escalado = Path(os.getcwd(), 'src', 'model', 'robust_scaler_model.pkl')

MODELOS = extraccion_modelos(ruta_modelos)
CLUSTER = extraccion_cluster(ruta_cluster)
ESCALER = extraccion_escalado(ruta_escalado)

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__, instance_relative_config=True)
app.config.from_object("config")

@app.route('/', methods = ['GET'])
def home():
    return render_template("home.html")

@app.route('/valorar')
def valorar():
    return render_template("valorar.html")


@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    try:
        
        if request.method == 'POST':

            balcon = 1 if request.form['balcon'] == "SI" else 0
            armarios = 1 if request.form['armarios'] == "SI" else 0
            estacionamiento = 1 if request.form['estacionamiento'] == "SI" else 0
            trastero = 1 if request.form['trastero'] == "SI" else 0
            terraza = 1 if request.form['terraza'] == "SI" else 0
            construccion = float(request.form['construccion'])
            metros = float(request.form['metros'])
            bannos = int(request.form['bannos'])
            lat = float(request.form['lat'])
            long = float(request.form['long'])
            cee = float(request.form['cee'])
            zona = float(request.form['zona'])
            tipo_inmueble = request.form['tipo_inmueble']

            piso, casa, duplex, atico = transf_tipo_inmueble(tipo_inmueble)

            construccion_e, metros_e = aplicar_escalado(construccion, metros, ESCALER)

            precio = aplicar_modelo(CLUSTER, MODELOS, balcon, armarios, estacionamiento, 
                                    trastero, terraza, construccion_e, metros_e, bannos, 
                                    lat, long, cee, zona, piso, casa, duplex, atico)
            
            return render_template("predict.html", precio_pred = f"€ {precio[0]:.2f}")
        else:
            return "Metodo no permitido"
    except ValueError:
        return render_template("error.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)