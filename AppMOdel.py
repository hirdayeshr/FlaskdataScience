from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load the Iris dataset
data = pd.read_csv("C:\\Users\\HIRDAYESH RAGHAV\\PycharmProjects\\pythonProject\\IRIS.csv")
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Load the saved model
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    # Make prediction
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

