from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the KNN model
knn_model = joblib.load('content/knn_model.pkl')

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        # Retrieve data from form as strings (for display)
        clump_thickness = request.form['clump_thickness']
        uniformity_cell_size = request.form['uniformity_cell_size']
        uniformity_cell_shape = request.form['uniformity_cell_shape']
        marginal_adhesion = request.form['marginal_adhesion']
        single_epithelial_cell_size = request.form['single_epithelial_cell_size']
        bland_chromatin = request.form['bland_chromatin']
        normal_nucleoil = request.form['normal_nucleoil']
        mitoses = request.form['mitoses']
        
        # Get selected classifier
        classifier = request.form.get('classifier', 'Decision Tree')  # Default to Decision Tree
        
        # Format the inputs to display them exactly as the user entered
        user_inputs = f"{clump_thickness}, {uniformity_cell_size}, {uniformity_cell_shape}, {marginal_adhesion}, {single_epithelial_cell_size}, {bland_chromatin}, {normal_nucleoil}, {mitoses}"

        # Only proceed if inputs are numeric (for prediction purposes)
        if (clump_thickness.isdigit() and uniformity_cell_size.isdigit() and 
            uniformity_cell_shape.isdigit() and marginal_adhesion.isdigit() and
            single_epithelial_cell_size.isdigit() and bland_chromatin.isdigit() and 
            normal_nucleoil.isdigit() and mitoses.isdigit()):

            # Convert the inputs to integers for model prediction
            input_data = np.array([[int(clump_thickness), int(uniformity_cell_size), int(uniformity_cell_shape),
                                    int(marginal_adhesion), int(single_epithelial_cell_size),
                                    int(bland_chromatin), int(normal_nucleoil), int(mitoses)]])
            
            # Predict using the loaded model
            prediction = knn_model.predict(input_data)

            # Format prediction result
            prediction_result = ''
            if prediction[0] == 2:
                prediction_result = "Benign"
            elif prediction[0] == 4:
                prediction_result = "Malignant"
        
        else:
            # If the inputs are not valid for prediction, default to an unknown state (skip prediction)
            prediction_result = "Cannot predict due to invalid input."

        # Return result to the template with user input and prediction
        return render_template('index.html',
                               inputs=user_inputs,
                               classifier=classifier,
                               prediction=prediction_result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
