import loadDataset
import model_CNN
import model_LSTM
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pyplot


# Set random states
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


'''
'  Function for dynamic learning rate
'''


def scheduler(epoch):
    if epoch < 5:
        return 0.001
    else:
        lr = 0.001 * tf.math.exp(0.1 * (5 - epoch))
        return lr.numpy()


'''
'  Configs
'''
epoch = 50
batchSize = 64
windowSize = [8, 16, 32, 64, 128, 256] # 32
models = ['CNN', 'LSTM']

'''
'  Main function
'''
for ws in windowSize:
    print('\n---- WindowSize = %d ----' % ws)
    # Load dataset
    print('---- Loading dataset ----')
    trainX, trainY, testX, testY = loadDataset.get_dataset(ws)

    # Compile model
    window_size, features, labels = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    for m in models:
        print('\nModelType: ' + m)
        if m == 'CNN':
            model = model_CNN.get_model(window_size, features, labels)
        if m == 'LSTM':
            model = model_LSTM.get_model(window_size, features, labels)
        print('---- Model compiled ----')

        # Train model
        print('---- Start training ----')
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
        history = model.fit(trainX,
                            trainY,
                            epochs=epoch,
                            batch_size=batchSize,
                            callbacks=[reduce_lr],
                            validation_data=(testX, testY),
                            verbose=1)
        print('----      Done      ----')

        # Get loss and accuracy of the model
        loss, accuracy = model.evaluate(testX, testY, verbose=2)

        # Clear plot
        pyplot.clf()

        # Plot Accuracy-Epochs figure
        pyplot.plot(history.history['accuracy'], label='Training accuracy')
        pyplot.plot(history.history['val_accuracy'], label='Validation accuracy')
        pyplot.title('Training and validation accuracy')
        pyplot.ylim([0.0, 1.0])
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Accuracy')
        pyplot.legend(loc='lower right')
        pyplot.savefig('./history/Training&validation_accuracy_' + m + '_' + str(ws) + '.png')

        # Clear plot
        pyplot.clf()

        # Plot Loss-Epochs figure
        pyplot.plot(history.history['loss'], label='Training loss')
        pyplot.plot(history.history['val_loss'], label='Validation loss')
        pyplot.title('Training and validation loss')
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Loss')
        pyplot.legend(loc='upper right')
        pyplot.savefig('./history/Training&validation_loss_' + m + '_' + str(ws) + '.png')

        # Save model
        model.save('./models/' + m + '_' + str(ws) + '.h5')
        print('---- Model saved ----')
