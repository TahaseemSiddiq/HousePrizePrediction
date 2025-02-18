from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs from the form
        features = [int(request.form["mainroad"]),
                    int(request.form["guestroom"]),
                    int(request.form["airconditioning"]),
                    int(request.form["prefarea"]),
                    int(request.form["furnishingstatus_unfurnished"]),
                    float(request.form["price_per_sqft"]),
                    int(request.form["total_rooms"]),
                    int(request.form["total_floors"]),
                    float(request.form["price_per_bedroom"])]
        
        # Convert input to NumPy array and reshape for prediction
        input_data = np.array([features]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction_text=f"Predicted House Price: ${prediction:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text="Error: Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
