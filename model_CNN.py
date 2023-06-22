from tensorflow import keras
'''
'  Method to compile a CNN model
'''


def get_model(window_size, features, labels, prt=False):
    # Define structure of the model
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(window_size, features)))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if prt:
        # Print out model structure
        model.summary()

    return model

