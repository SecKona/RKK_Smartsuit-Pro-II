from tensorflow import keras
'''
'  Method to compile a LSTM model
'''


def get_model(window_size, features, labels, prt=False):
    # Define structure of the model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=(window_size, features)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if prt:
        # Print out model structure
        model.summary()

    return model
