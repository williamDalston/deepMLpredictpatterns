import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Reshape, BatchNormalization, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

train_data = load_data('/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json')
train_solutions = load_data('/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json')
test_data = load_data('/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json')

import numpy as np
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess(data, solutions):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    X, y = [], []
    max_shape = (0, 0)
    mismatch_count = 0
    
    # First pass to find the maximum shape and check for mismatches
    for task_id, task in tqdm(data.items(), desc="Analyzing data"):
        if task_id not in solutions:
            logger.warning(f"Task ID {task_id} missing solutions, skipping...")
            continue
        
        data_pairs = len(task['train'])
        solution_pairs = len(solutions[task_id])
        
        if data_pairs != solution_pairs:
            logger.warning(f"Mismatch for Task ID {task_id}: {data_pairs} train pairs, {solution_pairs} solutions")
            mismatch_count += 1
        
        for pair in task['train']:
            input_array = np.array(pair['input'])
            max_shape = (max(max_shape[0], input_array.shape[0]),
                         max(max_shape[1], input_array.shape[1]))
    
    logger.info(f"Total mismatches: {mismatch_count}")
    logger.info(f"Maximum shape: {max_shape}")
    
    # Second pass to pad arrays to the maximum shape and gather the data
    for task_id, task in tqdm(data.items(), desc="Preprocessing data"):
        if task_id not in solutions:
            continue
        
        valid_pairs = min(len(task['train']), len(solutions[task_id]))
        
        for i in range(valid_pairs):
            try:
                input_array = np.array(task['train'][i]['input'])
                output_array = np.array(solutions[task_id][i])
                
                if input_array.ndim == 2 and output_array.ndim == 2:  # Check for 2D arrays
                    padded_input = np.pad(input_array, ((0, max_shape[0] - input_array.shape[0]), 
                                                        (0, max_shape[1] - input_array.shape[1])), 'constant')
                    padded_output = np.pad(output_array, ((0, max_shape[0] - output_array.shape[0]), 
                                                          (0, max_shape[1] - output_array.shape[1])), 'constant')
                    X.append(padded_input)
                    y.append(padded_output)
                else:
                    logger.warning(f"Invalid input/output shape for task {task_id}, pair {i}, skipping...")
            except Exception as e:
                logger.error(f"Unexpected error for Task ID {task_id}, pair {i}: {str(e)}")
                continue
    
    logger.info(f"Preprocessed {len(X)} valid input-output pairs")
    
    # Normalize data using StandardScaler or MinMaxScaler
    X = np.array(X)
    y = np.array(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y = scaler.transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
    
    return X, y, max_shape

# Load and preprocess data
train_data = load_data('/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json')
train_solutions = load_data('/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json')
X_train, y_train, max_shape = preprocess(train_data, train_solutions)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Add, Dropout, Activation, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

input_shape = (*max_shape, 1)
inputs = Input(shape=input_shape)

x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 32)
x = residual_block(x, 32)

x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 64)
x = residual_block(x, 64)

x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 128)
x = residual_block(x, 128)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(np.prod(input_shape), activation='sigmoid')(x)
outputs = Reshape(input_shape)(outputs)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

model.summary()

# Split data into training and validation sets
from sklearn.model_selection import train_test_split

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
history = model.fit(
    X_train_split, 
    y_train_split, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping, reduce_lr]
)

# Plotting training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error over epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.show()

# Ensure predict function works as intended
def predict(test_input, target_shape=None):
    input_array = np.array(test_input)
    target_shape = target_shape or X_train.shape[1:]
    if input_array.ndim != len(target_shape):
        raise ValueError("Input array dimensions do not match expected shape.")
    
    pad_width = [(0, max(0, dim - input_array.shape[i])) for i, dim in enumerate(target_shape)]
    input_array = np.pad(input_array, pad_width, 'constant')
    input_array = input_array.reshape(1, *input_array.shape, 1) 
    prediction = model.predict(input_array)
    return np.round(prediction.reshape(target_shape)).astype(int)

# Test the predict function with some examples
for i in range(5):
    test_input = X_train[i]
    predicted_output = predict(test_input, y_train[i].shape)
    print(f"Predicted Output {i}:\n{predicted_output}\nActual Output {i}:\n{y_train[i].reshape(y_train[i].shape)}\n")

# Ensure calculate_accuracy function works as intended
def calculate_accuracy(X, y):
    correct_predictions = 0
    total_predictions = len(X)
    for i in range(total_predictions):
        try:
            predicted_output = predict(X[i], y[i].shape)
        except ValueError as e:
            print(f"Warning: Skipping prediction for sample {i} due to shape mismatch: {e}")
            continue
        if np.array_equal(predicted_output, y[i]):
            correct_predictions += 1
    return correct_predictions / total_predictions

# Calculate accuracy on the training set
accuracy = calculate_accuracy(X_train, y_train)
print(f'Accuracy on training set: {accuracy * 100:.2f}%')
