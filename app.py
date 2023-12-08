from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

app = Flask(__name__)

df = pd.read_csv("drug_dataset_clean.csv")
X_train = df.drop("Drug", axis=1)
scaler.fit(X_train)

jenis_mapping = {
    0: 'Jenis obat A',
    1: 'Jenis obat B',
    2: 'Jenis obat C',
    3: 'Jenis obat X',
    4: 'Jenis obat Y',
    # Tambahkan sesuai kebutuhan
}


model = pickle.load(open('drug_models.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()
                      if x.replace('.', '', 1).isdigit()]
    scaled_features = scaler.transform([float_features])
    prediction = model.predict(scaled_features)

    result_text = ""
    if (prediction[0] == 0):
        result_text = "Jenis Obat Yang Digunakan DrugA"
    if (prediction[0] == 1):
        result_text = "Jenis Obat Yang Digunakan DrugB"
    if (prediction[0] == 2):
        result_text = "Jenis Obat Yang Digunakan DrugC"
    if (prediction[0] == 3):
        result_text = "Jenis Obat Yang Digunakan DrugX"
    if (prediction[0] == 4):
        result_text = "Jenis Obat Yang Digunakan DrugY"

    return render_template("index.html", prediction_text=result_text)


if __name__ == '__main__':
    app.run(debug=True)
