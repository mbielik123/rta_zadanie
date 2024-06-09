import numpy as np
from sklearn.linear_model import Perceptron
import joblib
from flask import Flask, request, jsonify

# Dane: cechy (X) i etykiety (y) dla funkcji XOR
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 1, 1, 0])


# Inicjalizacja i trening modelu perceptronowego
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
joblib.dump(clf, 'model.pkl')

# Użycie modelu do przewidzenia wyników dla tych samych danych
predictions = clf.predict(X)

# Wyświetlenie przewidywań i prawdziwych etykiet
print("Przewidywane etykiety:", predictions)
print("Prawdziwe etykiety:    ", y)

# Sprawdzenie, czy model się nauczył funkcji XOR
accuracy = np.mean(predictions == y)
print("Dokładność modelu:     ", accuracy)

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Przyjmujemy, że dane wejściowe są w formacie JSON i zawierają listę list
    features = np.array(data['features'])
    prediction = model.predict(features)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
