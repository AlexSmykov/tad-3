import tensorflow as tf
import numpy as np

import json
import os
import dvc.api
import warnings

VARIANTS = ['food', 'non_food']
IMAGE_SIZE = (256, 256)
IMAGE_KERNEL_SIZE = (256, 256, 1)
CLASSES_COUNT = len(VARIANTS)
BATCH_SIZE = 8
EPOCH_СOUNT = 5

dvc_params = dvc.api.params_show()
warnings.filterwarnings('ignore')

def create_layers(hidden_layers_count, hidden_layers_sizes, activation_functions, filters_count, dropout_part):
    layers = []

    layers.append(tf.keras.layers.Rescaling(scale=1. / 255))
    layers.append(tf.keras.layers.Conv2D(filters_count, (3, 3), activation='relu', input_shape=IMAGE_KERNEL_SIZE))
    layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
    layers.append(tf.keras.layers.Conv2D(filters_count, (3, 3), activation='relu'))
    layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
    layers.append(tf.keras.layers.Conv2D(filters_count, (3, 3), activation='relu'))
    layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
    layers.append(tf.keras.layers.Flatten())

    for i in range(hidden_layers_count):
        if activation_functions == "lelu":
            layers.append(tf.keras.layers.Dense(hidden_layers_sizes))
            layers.append(tf.keras.layers.LeakyReLU(0.2))
        else:    
            layers.append(tf.keras.layers.Dense(hidden_layers_sizes, activation=activation_functions))
        layers.append(tf.keras.layers.Dropout(dropout_part))
    
    layers.append(tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax'))
    
    return layers

models_parameters = [
    [2, 50, 'relu', 16, 0.12],
    [3, 150, 'lelu', 32, 0.15],
    [4, 500, 'relu', 64, 0.2],
]
SELECTED_MODEL = 0

selected_params = models_parameters[SELECTED_MODEL]

def get_grid(model, current_ds):
    grid = [[0, 0],
            [0, 0]]

    for test_pic_batch, test_val_batch in test_ds:
        predictions = model.predict(test_pic_batch, verbose=0)
        predicted_values = [np.argmax(prediction) for prediction in predictions]
        test_values = test_val_batch.numpy().tolist()

        for j in range(len(test_values)):
            grid[predicted_values[j]][test_values[j]] += 1 

    return grid

def get_prediction_accuracy(grid=None, model=None, current_ds=None):
    if not grid:
        grid = get_grid(model, current_ds)

    return float(grid[0][0] +  grid[1][1]) / float(grid[0][0] + grid[1][0] + grid[0][1] + grid[1][1]) 

def save_model(model, path): 
    try:
        os.mkdir('models')
    except:
        pass

    try:
        os.mkdir(f'models/{path}')
    except:
        pass

    model.save_weights(f'models/{path}/model')

print('Загрузка данных в модель')

ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    shuffle=True,
    seed=69,
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE)

ds_size = len(ds)
print(ds_size)

train_ds = ds.take(int(ds_size * 0.6))
validation_ds = ds.skip(int(ds_size * 0.6)).take(int(ds_size * 0.2))
test_ds = ds.skip(int(ds_size * 0.8))

print('Обучение модели')
print(f"Параметры: {selected_params}")

model = tf.keras.Sequential(create_layers(*selected_params))
model.compile('Adagrad', "sparse_categorical_crossentropy", ['accuracy'])
fit_result = model.fit(train_ds.cache(), validation_data=validation_ds.cache(), epochs=EPOCH_СOUNT)

grid = get_grid(model, test_ds)
save_model(model, SELECTED_MODEL)

accuracy = get_prediction_accuracy(get_grid(model, test_ds))
print(f'Точность обученной модели на тестовой выборке: {accuracy}')

with open('result.json', 'w') as f:
    json.dump({f'model {SELECTED_MODEL}': accuracy, 
               f'with_augmentation': dvc_params['with_augmentation']}, f)