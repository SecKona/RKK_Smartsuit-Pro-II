import loadDataset
import model_CNN
import model_LSTM
import numpy as np
import tensorflow as tf
from matplotlib import pyplot

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
# Set GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Use GPU 0
    gpu0 = gpus[0]
    # Set GPU RAM
    tf.config.experimental.set_memory_growth(gpu0, True)
    tf.config.set_visible_devices([gpu0], "GPU")

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
'  Method to train and evaluate a model
'''


def evaluate_model(train_x, train_y, test_x, test_y, model_type):
    # Compile model
    window_size, features, labels = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    if model_type == 'CNN':
        model = model_CNN.get_model(window_size, features, labels)
    if model_type == 'LSTM':
        model = model_LSTM.get_model(window_size, features, labels)

    # Train model
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.fit(train_x,
              train_y,
              epochs=epoch,
              batch_size=batchSize,
              callbacks=[reduce_lr],
              validation_data=(test_x, test_y),
              verbose=0)

    # Get loss and accuracy of the model
    loss, accuracy = model.evaluate(test_x, test_y, batch_size=batchSize, verbose=0)

    return loss, accuracy


'''
'  Method to summarize accuracy of models
'''


def summarize_results(scores, params, model_type):
    print('\n--- Summarize result ---\n')

    # Summarize AVG and STD
    for i in range(len(scores)):
        mean, std = np.mean(scores[i]), np.std(scores[i])
        print('Param=' + params[i] + ': %.3f%% (+/-%.3f)' % (mean, std))

    # Plot box plot
    pyplot.boxplot(scores, labels=params)
    pyplot.title(model_type + '_useAxisTest')
    pyplot.ylim([0, 100])
    pyplot.xlabel('Used axis')
    pyplot.ylabel('Accuracy (in %)')
    pyplot.savefig('./paramTestFigure/epoch=' + str(epoch) + '/' + model_type + '_useAxisTest.png')
    pyplot.show()


'''
'  Method to test window sizes
'''


def run_experiment(test_params, model_type, repeats=10):
    print('\n--- Start experiment ---\n')
    print('\nModelType: ' + model_type)
    # Repeat the experiment
    all_scores = list()
    for param in test_params:
        print('\n---- Used axis = ----' + param)
        # Load dataset
        train_x, train_y, test_x, test_y = loadDataset.get_dataset(32, axis=param)
        scores = list()
        print('\nRound: %d' % repeats)
        for r in range(repeats):
            loss, accuracy = evaluate_model(train_x, train_y, test_x, test_y, model_type)
            scores.append(accuracy * 100.0)
            print('%.3f%%' % (accuracy * 100.0))
        all_scores.append(scores)

    # Summarize the results
    summarize_results(all_scores, test_params, model_type)


'''
'  Configs
'''
epoch = 50
batchSize = 64
params = ['X', 'Y', 'Z', 'XYZ']

'''
'  Main function
'''
run_experiment(params, 'CNN')
run_experiment(params, 'LSTM')