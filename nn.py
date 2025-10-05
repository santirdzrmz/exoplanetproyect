import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# --------------------------
# 1. Load Data


NUM_CLASSES = 3   # 0: Not a possible exoplanet, 1: Candidate, 2: Confirmed Candidate
CLASS_LABELS = {0: "OTHER", 1: "CANDIDATE", 2: "CONFIRMED"}
RANDOM_SEED = 42

df = pd.read_csv("exoplanet_data_cleaned.csv")
NUM_FEATURES = df.shape[1]

# --------------------------
# 2. Data Preprocessing
# --------------------------

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("koi_disposition_encoded", axis = 1), df["koi_disposition_encoded"] , test_size = 0.2)
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)


# --------------------------
# 3. Define the MLP Model
# --------------------------

def create_mlp_model(input_shape, num_classes):
    """
    Defines a deep Multi-Layer Perceptron (MLP) model for multi-class classification.
    """
    model = Sequential([
        # Input layer automatically determined by the first Dense layer's input_shape
        Dense(256, activation='relu', input_shape=input_shape, name='dense_1_relu'),
        Dropout(0.3, name='dropout_1'), # Dropout for regularization
        
        Dense(128, activation='relu', name='dense_2_relu'),
        Dropout(0.3, name='dropout_2'),
        
        Dense(64, activation='relu', name='dense_3_relu'),
        Dropout(0.2, name='dropout_3'),
        
        # Output Layer: 3 neurons (one for each class) with Softmax activation.
        # Softmax ensures the output probabilities sum up to 1.
        Dense(num_classes, activation='softmax', name='output_softmax')
    ])
    return model

# Create and compile the model
mlp_model = create_mlp_model(input_shape=(NUM_FEATURES,), num_classes=NUM_CLASSES)

# Compile the model
mlp_model.compile(
    optimizer=Adam(learning_rate=0.001),
    # Use categorical_crossentropy for one-hot encoded multi-class labels
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("--- Model Architecture ---")
mlp_model.summary()
print("-" * 30 + "\n")

# --------------------------
# 4. Train the Model
# --------------------------

print("--- Training Model ---")
history = mlp_model.fit(
    X_train,
    y_train,
    epochs=100,             # Number of training iterations
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0              # Suppress output during training for cleaner demonstration
)
print(f"Training finished. Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print("-" * 30 + "\n")

# --------------------------
# 5. Make Predictions and Get Confidences
# --------------------------

# Take a few samples from the test set for prediction
sample_indices = [0, 1, 2]
X_new = X_test[sample_indices]
y_true = y_test[sample_indices]

# Predict probabilities (confidences) for the new samples
# .predict() outputs an array of probabilities for each class
confidence_predictions = mlp_model.predict(X_new, verbose=0)

print("--- Confidence Predictions Demonstration ---")

for i, preds in enumerate(confidence_predictions):
    true_label_idx = y_true[i]
    
    print(f"\nSample {i+1} (True Class: {CLASS_LABELS[true_label_idx]}):")
    
    # The predicted class is the one with the highest probability (argmax)
    predicted_class_idx = np.argmax(preds)
    predicted_class_label = CLASS_LABELS[predicted_class_idx]
    
    print(f"  > Predicted Class: {predicted_class_label} (Index: {predicted_class_idx})")
    print(f"  > Full Confidence Vector:")

    # Display the probability for each class
    for class_idx, class_label in CLASS_LABELS.items():
        confidence = preds[class_idx] * 100
        print(f"    - {class_label}: {confidence:.2f}%")

    # Optionally, print the raw feature vector (unscaled) for context
    print(f"  > Raw Feature Vector (first 5 features): {X_test[sample_indices[i]][:5]}")
    
print("-" * 30)

# --------------------------
# 6. Model Evaluation
# --------------------------

loss, accuracy = mlp_model.evaluate(X_test, y_test, verbose=0)
print(f"Model Final Test Loss: {loss:.4f}")
print(f"Model Final Test Accuracy: {accuracy:.4f}")