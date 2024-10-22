from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the models
knn_model = joblib.load('content/knn_model.pkl')
# nb_model = joblib.load('content/nb_model.pkl')  # Example for Naive Bayes
# dt_model = joblib.load('content/dt_model.pkl')  # Example for Decision Tree

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        # Retrieve data from form
        clump_thickness = int(request.form['clump_thickness'])
        uniformity_cell_size = int(request.form['uniformity_cell_size'])
        uniformity_cell_shape = int(request.form['uniformity_cell_shape'])
        marginal_adhesion = int(request.form['marginal_adhesion'])
        single_epithelial_cell_size = int(request.form['single_epithelial_cell_size'])
        bland_chromatin = int(request.form['bland_chromatin'])
        normal_nucleoil = int(request.form['normal_nucleoil'])
        mitoses = int(request.form['mitoses'])
        
        # Get selected classifier
        classifier = request.form.get('classifier', 'Nearest Neighbor')  # Default to Nearest Neighbor if not set
        
        # Prepare input for prediction
        input_data = np.array([[clump_thickness, uniformity_cell_size, uniformity_cell_shape,
                                marginal_adhesion, single_epithelial_cell_size,
                                bland_chromatin, normal_nucleoil, mitoses]])
        
        # Predict using the selected model
        prediction = knn_model.predict(input_data)

        # Format prediction result
        prediction_result = ''
        if prediction[0] == 2:
            prediction_result = "Benign"
        elif prediction[0] == 4:
            prediction_result = "Malignant"
        
        # Return result to the template
        return render_template('index.html',
                               inputs=f"{clump_thickness}, {uniformity_cell_size}, {uniformity_cell_shape}, "
                                      f"{marginal_adhesion}, {single_epithelial_cell_size}, "
                                      f"{bland_chromatin}, {normal_nucleoil}, {mitoses}",
                               classifier=classifier,
                               prediction=prediction_result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
