from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# Importar los modelos
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
en = pickle.load(open('encoder.pkl','rb'))

# crear flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    G = request.form['Genero']
    A = request.form['Edad']
    Ai = request.form['ingreso']
    SS = request.form['gasto']
    

    feature_list = ([[G,A,Ai,SS]])
 
    transformed_features = en.transform(feature_list)
    transformed_features[:,2:] = sc.transform(transformed_features[:,2:])
    prediction = model.predict(transformed_features).reshape(1,-1)

    diccionario = {0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5"}

    if prediction[0][0] in diccionario:
        cliente = diccionario[prediction[0][0]]
        result =("El usuario pertenece al : {} ".format(cliente))
    else:
        result =("No se pudo agrupar al cliente")
    return render_template('index.html',result = result)

# python main
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=False)
