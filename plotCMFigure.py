from matplotlib import pyplot
from numpy import argmax
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

import loadDataset

'''
'  Method to plot confusion matrix
'''


def plot_cm_figure(dataset_x, dataset_y, model_type, window_size, save_fig=True):
    # Clear plot
    pyplot.clf()

    # Dictionary of labels
    label_mapping = {0: "Jogging", 1: "Jumping", 2: "Standing", 3: "Walking", 4: "Laying", 5: "Sitting"}

    # Load model
    model = load_model('./models/' + model_type + '_' + str(window_size) + '.h5')

    # Print accuracy & loss
    loss, accuracy = model.evaluate(dataset_x, dataset_y, verbose=2)

    # Predict dataset
    predictions = model.predict(dataset_x, verbose=0)
    predicted_labels = argmax(predictions, axis=1)
    true_labels = argmax(dataset_y, axis=1)

    # Get confusion matrix
    tmp_confusion_matrix = confusion_matrix(true_labels, predicted_labels)

    heatmap(tmp_confusion_matrix,
            annot=True,
            fmt='d',
            linewidth=1,
            cmap='Blues',
            xticklabels=list(label_mapping.values()),
            yticklabels=list(label_mapping.values()))

    pyplot.xlabel('Predicted Labels')
    pyplot.ylabel('True Labels')

    if save_fig:
        # Output figure
        pyplot.savefig('./confusionMatrices/Figure_confusion_matrix_' + model_type + '_' + str(window_size) + '.png')
        print('----  Figure saved  ----')
    # pyplot.show()


'''
'  Config
'''
windowSize = [8, 16, 32, 64, 128, 256]
# windowSize = [32]

'''
'  Main function
'''
for ws in windowSize:
    print('\n---- WindowSize = %d ----' % ws)

    # Load dataset
    print('---- Loading dataset ----')
    trainX, trainY, testX, testY = loadDataset.get_dataset(ws)

    # Plot confusion matrix
    plot_cm_figure(testX, testY, 'CNN', ws)
    plot_cm_figure(testX, testY, 'LSTM', ws)
