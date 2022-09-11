from flask import Flask, request, render_template

import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = request.form['sepal_length']
        sepal_width = request.form['sepal_width']
        petal_length = request.form['petal_length']
        petal_width = request.form['petal_width']
    except:
        return render_template('error.html', err='Missing argument!')

    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    if (not isfloat(sepal_length)) or (not isfloat(sepal_width)) or (not isfloat(petal_length)) or (not isfloat(petal_width)):
        return render_template('error.html', err='Type error!')

    filename = 'iris_SVC_model.z'
    loaded_model = joblib.load(filename)

    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]]).reshape(1, -1)

    prediction = loaded_model.predict(x)

    return render_template('result.html', sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width,prediction=prediction[0])

if __name__ == '__main__':
    app.run('localhost', 5000, debug=True)