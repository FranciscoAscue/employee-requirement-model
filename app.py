from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sklearn
import pickle

# Importar los modelos
rf_classifier = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
 
# crear flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    sl_no = request.form['sl_no']
    gender = request.form['Genero']  

    ssc_p = float(request.form['ssc_p'])
    hsc_p = float(request.form['hsc_p'])
    degree_p = float(request.form['degree_p'])
    
    workex = request.form['workex']
    
    etest_p = float(request.form['etest_p'])
    
    specialisation = request.form['specialisation']
    
    mba_p = float(request.form['mba_p'])

    data = [[sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p]]
    columns = ['sl_no', 'gender', 'ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p']
    df = pd.DataFrame(data, columns=columns)
    specialisation_mapping = {'Mkt&HR': 0, 'Mkt&Fin': 1}
    #status_mapping = {'Placed': 0, 'Not Placed': 1}
    workex_mapping = {'No': 0, 'Yes': 1}
    gender_mapping = {'M': 0, 'F': 1}
    
    df['specialisation'] = df['specialisation'].map(specialisation_mapping)
    df['workex'] = df['workex'].map(workex_mapping)
    df['gender'] = df['gender'].map(gender_mapping)
    df_t = scaler.transform(df)
    prediction = rf_classifier.predict(df_t)

    if prediction == 0:
        result = 'contratado'
    else:
        result = 'No Contratado'

    return render_template('index.html', result=result)
# python main
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=False)
