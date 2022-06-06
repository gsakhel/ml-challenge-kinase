from flask import Flask, render_template, request, Markup
from rdkit import Chem
from mordred import Calculator, descriptors 
import pandas as pd
import numpy as np
import torch
import pickle

app  = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template("index.html")

    
def predictor(input_smile):
    # Load files
    chem_calc = Calculator(descriptors, ignore_3D=True)
    model_columns = pd.read_csv('model_columns.csv').columns
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model = torch.jit.load('model_scripted.pt')

    # Prepare input
    mol = Chem.MolFromSmiles(input_smile)
    input_features = chem_calc.pandas([mol])
    input_features = pd.concat([input_features]*4)
    input_features[['JAK1', 'JAK2', 'JAK3', 'TYK2']] = np.diag([1,1,1,1])
    input_features[['measurement_type_pIC50', 'measurement_type_pKi']] = [0,1]
    input_features = input_features[model_columns]
    scaled_input_features = pd.DataFrame(scaler.transform(input_features), columns=model_columns)
    scaled_input_features_tensor = torch.tensor(scaled_input_features.values)

    # Predict
    model.eval()
    with torch.no_grad():
        result = model(scaled_input_features_tensor)
    
    return result

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        input = request.form.to_dict()['smile']
        result = torch.flatten(predictor(input))
    return render_template("result.html", jak1=result[0].item(), jak2=result[1].item(), jak3=result[2].item(), tyk2=result[3].item())

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001)
