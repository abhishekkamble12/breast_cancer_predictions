from flask import Flask, render_template, request, url_for
import numpy as np
import joblib 

app = Flask(__name__)

# Keep only the features we're using in the form
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form
        features_list = []
        for name in FEATURE_NAMES:
            value = request.form.get(name)
            if value is None:
                raise ValueError(f"Missing field: {name}")
            features_list.append(float(value))
        
        # Convert to numpy array and reshape
        final_features = np.array(features_list).reshape(1, -1)
        
        # Make prediction (assuming model is loaded correctly)
        prediction = model.predict(final_features)
        
        # Format output
        if prediction[0] == 1:
            prediction_text = 'The tumor is Malignant (Cancerous).'
            prediction_class = 'malignant'
        else:
            prediction_text = 'The tumor is Benign (Not Cancerous).'
            prediction_class = 'benign'
            
        return render_template('index.html', 
                             prediction_text=prediction_text,
                             prediction_class=prediction_class)
                             
    except Exception as e:
        return render_template('index.html',
                             prediction_text=f"Error: {str(e)}",
                             prediction_class='malignant')

if __name__ == "__main__":
    # Load model when starting the application
    try:
        model = joblib.load('model.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
    app.run(debug=True)
