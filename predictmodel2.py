import numpy as np
import json
import os
import matplotlib.pyplot as plt

# Load dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

train_data = load_data('/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json')
train_solutions = load_data('/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json')
test_data = load_data('/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json')

# Function to visualize data
def visualize_data(X, y, index=0):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(X[index].reshape(X.shape[1], X.shape[2]), cmap='gray')
    axes[0].set_title("Input")
    axes[1].imshow(y[index].reshape(y.shape[1], y.shape[2]), cmap='gray')
    axes[1].set_title("Output")
    plt.show()
import numpy as np
import logging
from tqdm import tqdm

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
    return np.array(X), np.array(y), max_shape

# Sample usage
X_train, y_train, max_shape = preprocess(train_data, train_solutions)

# Reshape, normalize, and add a channel dimension
X_train = X_train.reshape(-1, *max_shape, 1) / 9.0
y_train = y_train.reshape(-1, *max_shape, 1) / 9.0

# Visualize the first sample in your training data
visualize_data(X_train, y_train, index=0)
Analyzing data: 100%|██████████| 400/400 [00:00<00:00, 10280.60it/s]
Preprocessing data: 100%|██████████| 400/400 [00:00<00:00, 4995.52it/s]

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Add, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def residual_block(x, filters, kernel_size=3, stride=1, activation='relu'):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    return x

input_shape = X_train.shape[1:]
inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 64, stride=2)
x = residual_block(x, 128, stride=2)
x = residual_block(x, 256, stride=2)
x = residual_block(x, 512, stride=2)

x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(np.prod(input_shape), activation='sigmoid')(x)
outputs = Reshape(input_shape)(outputs)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train with validation and early stopping
history = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)

# Plotting training history
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
2024-07-22 07:25:28.193251: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-07-22 07:25:28.193389: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-07-22 07:25:28.354238: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[3], line 40
     38 x = Dropout(0.5)(x)
     39 outputs = Dense(np.prod(input_shape), activation='sigmoid')(x)
---> 40 outputs = Reshape(input_shape)(outputs)
     42 model = Model(inputs, outputs)
     43 model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

NameError: name 'Reshape' is not defined
def predict(test_input, target_shape=None):
    input_array = np.array(test_input)
    target_shape = target_shape or X_train.shape[1:]
    if input_array.ndim != len(target_shape):
        raise ValueError("Input array dimensions do not match expected shape.")
    
    pad_width = [(0, max(0, dim - input_array.shape[i])) for i, dim in enumerate(target_shape)]
    input_array = np.pad(input_array, pad_width, 'constant')
    input_array = input_array.reshape(1, *input_array.shape, 1) / 9.0 
    prediction = model.predict(input_array)
    return np.round(prediction.reshape(target_shape) * 9).astype(int)

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
# Predict function for test input with debugging and robust shape handling
def predict(test_input, target_shape=None):
    input_array = np.array(test_input)
    target_shape = target_shape or X_train.shape[1:]  # Default to X_train shape
    if input_array.ndim != len(target_shape):  # Ensure compatible dimensions
        raise ValueError("Input array dimensions do not match expected shape.")

    pad_width = [(0, max(0, dim - input_array.shape[i])) for i, dim in enumerate(target_shape)]
    input_array = np.pad(input_array, pad_width, 'constant')
    input_array = input_array.reshape(1, *input_array.shape, 1) / 9.0 
    prediction = model.predict(input_array)
    return np.round(prediction.reshape(target_shape) * 9).astype(int)

# Function to calculate accuracy
def calculate_accuracy(X, y):
    correct_predictions = 0
    total_predictions = len(X)
    for i in range(total_predictions):
        try:
            predicted_output = predict(X[i], y[i].shape)  # Pass target shape
        except ValueError as e:
            print(f"Warning: Skipping prediction for sample {i} due to shape mismatch: {e}")
            continue
        if np.array_equal(predicted_output, y[i]):
            correct_predictions += 1
    return correct_predictions / total_predictions

# Calculate accuracy on the training set
accuracy = calculate_accuracy(X_train, y_train)
print(f'Accuracy on training set: {accuracy * 100:.2f}%')