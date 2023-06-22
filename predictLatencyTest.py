import loadDataset
import numpy as np
import time
from matplotlib import pyplot
from tensorflow import keras

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

'''
'  Method to get predict latency
'''


def get_predict_latency(dataset_x, model_type, window_size):
    # Load model
    model = keras.models.load_model('./models/' + model_type + '_' + str(window_size) + '.h5')

    # Predict dataset
    start = time.perf_counter()
    predictions = model.predict(dataset_x, verbose=0)
    end = time.perf_counter()

    return end - start


'''
'  Method to summarize latency of models
'''


def summarize_results(scores, params, model_type):
    print('\n--- Summarize result ---\n')

    # Summarize AVG and STD
    for i in range(len(scores)):
        mean, std = np.mean(scores[i]), np.std(scores[i])
        print('Param=%d: %.3f (+/-%.3f)' % (params[i], mean, std))

    # Plot box plot
    pyplot.boxplot(scores, labels=params)
    pyplot.title('PredictLatencyTest_' + model_type)
    pyplot.xlabel('Window size (in frame)')
    pyplot.ylabel('Predict time (in s)')

    # Output figure
    pyplot.savefig('./paramTestFigure/Figure_predict_latency_' + model_type + '.png')
    print('----  Figure saved  ----')
    pyplot.show()


'''
'  Method to test latency
'''


def run_experiment(window_size, model_type, repeats=10):
    print('\n--- Start experiment ---\n')
    # Repeat the experiment
    all_scores = list()
    for ws in window_size:
        print('\n---- WindowSize = %d ----' % ws)
        # Load dataset
        print('---- Loading dataset ----')
        train_x, train_y, test_x, test_y = loadDataset.get_dataset(ws)

        # Plot confusion matrix
        scores = list()
        for r in range(repeats):
            latency = get_predict_latency(test_x, model_type, ws)
            scores.append(latency)
            print('%.3f' % latency)

        all_scores.append(scores)

    # Summarize the results
    summarize_results(all_scores, window_size, model_type)


'''
'  Config
'''
window_sizes = [8, 16, 32, 64, 128, 256]

'''
'  Main function
'''
run_experiment(window_sizes, 'CNN')
run_experiment(window_sizes, 'LSTM')
