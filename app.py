import os
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
import tensorflow as tf

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# Define directories
MODEL_DIR = os.getenv('MODEL_DIR', r'E:\projects\Heart_Attack_Prediction\path_to_save_models')
PREPROCESSOR_PATH = os.getenv('PREPROCESSOR_PATH', r'E:\projects\Heart_Attack_Prediction\models\preprocessor.pkl')

# Function to load a model or preprocessor
def load_model(filepath):
    try:
        if filepath.endswith('.keras') or filepath.endswith('.h5'):
            model = tf.keras.models.load_model(filepath)
        else:
            model = joblib.load(filepath)
        logging.info(f"{os.path.basename(filepath)} loaded successfully. Type: {type(model)}")
        return model
    except Exception as e:
        logging.error(f"Error loading {os.path.basename(filepath)}: {e}")
        return None

# Load models dynamically
model_files = [
    'adaboost.joblib', 'bagging_classifier.joblib', 'bernoulli_nb.joblib', 'boosting_classifier.joblib',
    'extra_trees.joblib', 'gaussian_nb.joblib',  'hist_gradient_boosting.joblib',
    'lasso_logistic.joblib', 'lda.joblib', 'lightgbm.joblib', 'logistic_regression.joblib',
    'random_forest.joblib', 'stacking_classifier.joblib', 'svc.joblib',
    'voting_classifier_hard.joblib', 'voting_classifier_soft.joblib', 'xgboost.joblib'
]

models = {}
for model_file in model_files:
    model_name = os.path.splitext(model_file)[0]
    models[model_name] = load_model(os.path.join(MODEL_DIR, model_file))

# Load preprocessor
preprocessor = load_model(PREPROCESSOR_PATH)

if preprocessor is None or not isinstance(preprocessor, ColumnTransformer):
    logging.error("Preprocessor failed to load or is not valid.")
    raise RuntimeError("Preprocessor is not valid.")

# Initialize Flask app
app = Flask(__name__)

# Define valid input ranges and categories
valid_ranges = {
    'age': (29, 120),
    'trtbps': (50, 250),
    'chol': (50, 500),
    'thalachh': (60, 220),
    'oldpeak': (0, 6),
    'caa': (0, 3)
}

valid_categories = {
    'sex': {0, 1},
    'cp': {0, 1, 2, 3},
    'fbs': {0, 1},
    'restecg': {0, 1, 2},
    'exng': {0, 1},
    'slp': {0, 1, 2},
    'thall': {0, 1, 2, 3}
}

expected_features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
                     'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']

def validate_input(form_data):
    """Validate input ranges and categories."""
    for key, (min_val, max_val) in valid_ranges.items():
        if key in form_data and not (min_val <= form_data[key] <= max_val):
            raise ValueError(f"{key} must be between {min_val} and {max_val}")
    for key, valid_values in valid_categories.items():
        if key in form_data and form_data[key] not in valid_values:
            raise ValueError(f"{key} must be one of {valid_values}")

# Function to generate a bar chart for feature importances
def plot_feature_importances(importances, feature_names, model_name):
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Feature Importances')
    plt.tight_layout()
    
    # Save the plot to a BytesIO object to pass to Flask
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    
    img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            logging.info(f"Form Data Received: {request.form}")

            # Extract and validate form data
            form_data = {
                'age': float(request.form.get('age', 0)),
                'sex': int(request.form.get('sex', 0)),
                'cp': int(request.form.get('cp', 0)),
                'trtbps': float(request.form.get('trtbps', 0)),
                'chol': float(request.form.get('chol', 0)),
                'fbs': int(request.form.get('fbs', 0)),
                'restecg': int(request.form.get('restecg', 0)),
                'thalachh': float(request.form.get('thalachh', 0)),
                'exng': int(request.form.get('exng', 0)),
                'oldpeak': float(request.form.get('oldpeak', 0)),
                'slp': int(request.form.get('slp', 0)),
                'caa': int(request.form.get('caa', 0)),
                'thall': int(request.form.get('thall', 0))
            }

            # Validate input
            validate_input(form_data)

            logging.info(f"Validated Form Data: {form_data}")

            # Create a DataFrame from input
            input_data = pd.DataFrame([form_data])
            input_data = input_data.reindex(columns=expected_features, fill_value=0)

            # Preprocess data
            X_transformed = preprocessor.transform(input_data)

            # Make predictions and generate explanations
            predictions = {}
            explanations = {}  # To hold feature importance or explanations for each model
            feature_importance_images = {}  # To store base64 image strings for visualizations
            for model_name, model in models.items():
                if model:
                    pred = model.predict(X_transformed)[0]  # Get the prediction (likely a 0 or 1)
                    predictions[model_name] = "Yes" if pred == 1 else "No"
                    
                    # Explanation part (for models like random forest, XGBoost, etc.)
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_explanation = dict(zip(expected_features, importances))
                        explanations[model_name] = feature_explanation

                        # Generate visualization of feature importance
                        feature_importance_images[model_name] = plot_feature_importances(importances, expected_features, model_name)
                    elif hasattr(model, 'coef_'):  # For models like Logistic Regression
                        coef = model.coef_[0]
                        feature_explanation = dict(zip(expected_features, coef))
                        explanations[model_name] = feature_explanation

                        # Visualization of coefficients
                        feature_importance_images[model_name] = plot_feature_importances(coef, expected_features, model_name)
                    else:
                        explanations[model_name] = "No explanation available"

            logging.info(f"Predictions: {predictions}")
            logging.info(f"Explanations: {explanations}")

            # Return predictions, explanations, and images to the result template
            return render_template('result.html', predictions=predictions, explanations=explanations, 
                                   feature_importance_images=feature_importance_images)

        except ValueError as ve:
            logging.error(f"Value error: {ve}")
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

if __name__ == '__main__':
    app.run(debug=True)
