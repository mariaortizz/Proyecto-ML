'''Archivo por el que corre la aplicación'''

from flask import Flask
import os 

os.chdir(os.path.dirname(__file__))

app = Flask(__name__, instance_relative_config=True)
# app.config.from_object('config.py')
app.config.from_pyfile(r'..\config.py')

@app.route('/')
def index():
    # print(app.config['VARIABLE'])
    print(app.config(['VARIBLE2']))
    return "<h1>Hola Mundoooooooooo</h1>"

if __name__ == "__main__":
    app.run(port=5000, debug=True)