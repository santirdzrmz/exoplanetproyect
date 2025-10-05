
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# --- Input Variables ---
# The following variables are expected in the input CSV file.
# These are the features used to train the XGBoost model.
INPUT_VARIABLES = [
    'koi_period', 
    'koi_time0bk', 
    'koi_impact', 
    'koi_duration', 
    'koi_depth', 
    'koi_prad', 
    'koi_teq', 
    'koi_insol', 
    'koi_model_snr', 
    'koi_srad'
]

def preprocess_data(df):
    """
    Preprocesses the input DataFrame to be used by the XGBoost model.
    This includes handling missing values and selecting the correct features.
    """
    # Drop rows with missing values
    df = df.dropna(subset=INPUT_VARIABLES)
    
    # Select only the input variables
    X = df[INPUT_VARIABLES]
    
    return X

def predict(input_csv_path, model_path="best_xgb_model.joblib"):
    """
    Loads a pre-trained XGBoost model and uses it to predict the class of exoplanets
    from a new CSV file.

    Args:
        input_csv_path (str): The path to the new CSV file.
        model_path (str): The path to the pre-trained XGBoost model.

    Returns:
        pandas.DataFrame: A DataFrame with the original data and two new columns:
                          'predicted_class' and 'predicted_probability'.
    """
    # Load the new data
    new_data = pd.read_csv(input_csv_path)

    # Preprocess the data
    X_new = preprocess_data(new_data.copy())

    # Load the pre-trained model
    model = joblib.load(model_path)

    # Check if the model was loaded successfully
    if model is None:
        raise FileNotFoundError(f"Could not load model from {model_path}. Please check the path.")

    # Make predictions
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    # Add the predictions to the new_data DataFrame
    new_data['predicted_class'] = predictions
    new_data['predicted_probability'] = [max(prob) for prob in probabilities]
    
    # Map the predicted class to the original labels
    class_map = {2: 'CONFIRMED', 1: 'CANDIDATE', 0: 'OTHER'}
    new_data['predicted_class'] = new_data['predicted_class'].map(class_map)

    return new_data

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict exoplanet class using a pre-trained XGBoost model.')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file.')
    parser.add_argument('--model_path', type=str, default='best_xgb_model.joblib', help='Path to the pre-trained XGBoost model.')
    parser.add_argument('--output_csv', type=str, default='predictions.csv', help='Path to save the output CSV file.')

    args = parser.parse_args()

    # Make predictions
    predictions_df = predict(args.input_csv, args.model_path)

    # Save the predictions to a new CSV file
    predictions_df.to_csv(args.output_csv, index=False)

    print(f"Predictions saved to {args.output_csv}")
