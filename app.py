from flask import Flask, render_template, request, Markup
from rdkit import Chem
from mordred import Calculator, descriptors 
import pandas as pd
import numpy as np
import torch
import pickle
from rdkit.Chem.Draw import rdMolDraw2D

app  = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template("index.html")

    
def predictor(input_smile):
    """Predicts pKi's for 4 inhibitors given SMILE

    Args:
    input_smile:  molecular structure in SMILE format

    Output:
    result:       tensor with pKi's
    
    """

    # Load files
    chem_calc = Calculator(descriptors, ignore_3D=True)
    model_columns = pd.read_csv('model_assets/model_columns.csv').columns
    scaler = pickle.load(open("model_assets/scaler.pkl", "rb"))
    model = torch.jit.load('model_assets/model_scripted.pt')

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

def draw_molecule(mol, molSize=(450, 150), kekulize=True):
    """Draw a molecule in SVG format

    Args:
    mol:       molecule of interest in SMILE format
    molSize:   size of output SVG
    kekulize:  make double bonds explicit

    Output:
    svg:       SVG of our drawn molecule

    """
    
    molecule = Chem.MolFromSmiles(mol)
    if kekulize:
        try:
            Chem.Kekulize(molecule)
        except:
            molecule = Chem.Mol(mol.ToBinary())
    if not molecule.GetNumConformers():
        Chem.rdDepictor.Compute2DCoords(molecule)
    
    painter = rdMolDraw2D.MolDraw2DSVG(*molSize)
    painter.DrawMolecule(molecule)
    painter.FinishDrawing()
    svg = painter.GetDrawingText().replace('svg:', '')
    return svg


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        input = request.form.to_dict()['smile']
        result = torch.flatten(predictor(input))
        drawing = draw_molecule(input)
        print("SUCCESS!")
    return render_template("result.html", jak1=result[0].item(), jak2=result[1].item(), jak3=result[2].item(), tyk2=result[3].item(), drawing=Markup(drawing))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
