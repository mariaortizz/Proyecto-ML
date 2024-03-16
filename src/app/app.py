'''Archivo por el que corre la aplicación'''

from flask import Flask
import os 

os.chdir(os.path.dirname(__file__))

app = Flask(__name__, instance_relative_config=True)
app.config.from_object("config")
# app.config.from_pyfile("config.py")

@app.route('/')
def index():
    print(app.config['VARIABLE'])
    return "<h1>Hola Mundoooooooooo</h1>"

# @app.route('/user/<name>/<int:index>')
# def funcion_0(name, index):
#     my_list = ['e1', 'e2', 'e3', 'e4']
#     my_dict = {'k1':'v1', 'k2':'v2'}
#     my_tuple = ('t1', 't2')
#     return render_template('test.html', name = name, myindex = index, mylist = my_list, 
#                            mydict = my_dict, mytuple = my_tuple)
if __name__ == "__main__":
    app.run(port=5000, debug=True)