import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

'''
'  Method to load dataset from CSV file as pandas.DataFrame
'  1) file_path='./csv_original/'
'  2) file_path='./csv_balanced/'
'''


def load_csv(file_name, file_path='./csv_original/', axis='XYZ', part='All'):
    # Load CSV file with given file name and column name
    dataset = pd.read_csv(file_path + file_name, header=0)

    dataset = dataset[['Activity', 'Time',
                       'Head.X', 'Head.Y', 'Head.Z',
                       'Neck.X', 'Neck.Y', 'Neck.Z',
                       'Hips.X', 'Hips.Y', 'Hips.Z',

                       'LeftShoulder.X', 'LeftShoulder.Y', 'LeftShoulder.Z',
                       'LeftArm.X', 'LeftArm.Y', 'LeftArm.Z',
                       'LeftForeArm.X', 'LeftForeArm.Y', 'LeftForeArm.Z',
                       'LeftHand.X', 'LeftHand.Y', 'LeftHand.Z',
                       'LeftThigh.X', 'LeftThigh.Y', 'LeftThigh.Z',
                       'LeftShin.X', 'LeftShin.Y', 'LeftShin.Z',
                       'LeftFoot.X', 'LeftFoot.Y', 'LeftFoot.Z',
                       'LeftToe.X', 'LeftToe.Y', 'LeftToe.Z',
                       'LeftToeTip.X', 'LeftToeTip.Y', 'LeftToeTip.Z',

                       'RightShoulder.X', 'RightShoulder.Y', 'RightShoulder.Z',
                       'RightArm.X', 'RightArm.Y', 'RightArm.Z',
                       'RightForeArm.X', 'RightForeArm.Y', 'RightForeArm.Z',
                       'RightHand.X', 'RightHand.Y', 'RightHand.Z',
                       'RightThigh.X', 'RightThigh.Y', 'RightThigh.Z',
                       'RightShin.X', 'RightShin.Y', 'RightShin.Z',
                       'RightFoot.X', 'RightFoot.Y', 'RightFoot.Z',
                       'RightToe.X', 'RightToe.Y', 'RightToe.Z',
                       'RightToeTip.X', 'RightToeTip.Y', 'RightToeTip.Z'
                       ]]

    # Split out timestamps array
    dataset.pop('Time')

    # Split out height of hips as new feature column
    hips_height = dataset[['Hips.Y']]
    hips_height.columns = ['Hips_height']
    hips_height = hips_height.drop(0)

    # Split out the label array
    dataset_y = dataset.pop('Activity')
    dataset_y = dataset_y.drop(0)

    if axis == 'X':
        dataset = dataset.iloc[:, dataset.columns.str.endswith('.X')]

    if axis == 'Y':
        dataset = dataset.iloc[:, dataset.columns.str.endswith('.Y')]

    if axis == 'Z':
        dataset = dataset.iloc[:, dataset.columns.str.endswith('.Z')]

    if part != 'All':
        dataset = dataset.iloc[:, dataset.columns.str.contains(part)]

    # Convert into relative movement
    dataset = dataset.diff()
    dataset = dataset.drop(0)

    # Merge feature column
    dataset = dataset.join(hips_height)

    return dataset, dataset_y


'''
'  Method to slice time series with given window size and overlap rate
'''


def sliding_window(dataset_x, dataset_y, window_size, slide_step=2, overlap_rate=0.8):
    if len(dataset_x) < window_size:
        print('[ERROR]Cannot perform slidingwindow since input sequence is too short!')
        return

    start_index = 0
    while start_index in range(len(dataset_x)):
        end_index = start_index + window_size

        if end_index < len(dataset_x):
            # Slice data window
            temp_x = dataset_x[start_index: end_index]
            # Reshape data window
            temp_x = temp_x.values.reshape(1, window_size, temp_x.shape[1])
            temp_y = dataset_y[start_index: end_index].mode()

            if start_index == 0:
                output_x = temp_x
                output_y = temp_y
            else:
                output_x = np.concatenate((output_x, temp_x), axis=0)
                output_y = np.concatenate((output_y, temp_y), axis=0)

        # start_index += int(window_size * (1 - overlap_rate))
        start_index += slide_step

    return output_x, output_y


'''
'  Method to load all CSV file as dataset
'''


def get_dataset(window_size, axis='XYZ', part='All', if_plot=False, prt_shape=True):
    # Load CSV file
    train_jog_x, train_jog_y = load_csv("Jogging_worldpos.csv", axis=axis, part=part)
    train_jump_x, train_jump_y = load_csv("Jumping_worldpos.csv", axis=axis, part=part)
    train_stand_x, train_stand_y = load_csv("Standing_worldpos.csv", axis=axis, part=part)
    train_walk_x, train_walk_y = load_csv("Walking_worldpos.csv", axis=axis, part=part)
    train_lay_x, train_lay_y = load_csv("Laying_worldpos.csv", axis=axis, part=part)
    train_sit_x, train_sit_y = load_csv("Sitting_worldpos.csv", axis=axis, part=part)

    test_jump_x, test_jump_y = load_csv("Jogging_1_worldpos.csv", axis=axis, part=part)
    test_jog_x, test_jog_y = load_csv("Jumping_1_worldpos.csv", axis=axis, part=part)
    test_stand_x, test_stand_y = load_csv("Standing_1_worldpos.csv", axis=axis, part=part)
    test_walk_x, test_walk_y = load_csv("Walking_1_worldpos.csv", axis=axis, part=part)
    test_lay_x, test_lay_y = load_csv("Laying_1_worldpos.csv", axis=axis, part=part)
    test_sit_x, test_sit_y = load_csv("Sitting_1_worldpos.csv", axis=axis, part=part)

    # Perform slidingwindow approach
    train_jog_x, train_jog_y = sliding_window(train_jog_x, train_jog_y, window_size)
    train_jump_x, train_jump_y = sliding_window(train_jump_x, train_jump_y, window_size)
    train_stand_x, train_stand_y = sliding_window(train_stand_x, train_stand_y, window_size)
    train_walk_x, train_walk_y = sliding_window(train_walk_x, train_walk_y, window_size)
    train_lay_x, train_lay_y = sliding_window(train_lay_x, train_lay_y, window_size)
    train_sit_x, train_sit_y = sliding_window(train_sit_x, train_sit_y, window_size)

    test_jump_x, test_jump_y = sliding_window(test_jump_x, test_jump_y, window_size)
    test_jog_x, test_jog_y = sliding_window(test_jog_x, test_jog_y, window_size)
    test_stand_x, test_stand_y = sliding_window(test_stand_x, test_stand_y, window_size)
    test_walk_x, test_walk_y = sliding_window(test_walk_x, test_walk_y, window_size)
    test_lay_x, test_lay_y = sliding_window(test_lay_x, test_lay_y, window_size)
    test_sit_x, test_sit_y = sliding_window(test_sit_x, test_sit_y, window_size)

    # Merge dataset_x and dataset_y
    train_x = np.concatenate((train_jog_x, train_jump_x, train_stand_x, train_walk_x, train_lay_x, train_sit_x), axis=0)
    train_y = np.concatenate((train_jog_y, train_jump_y, train_stand_y, train_walk_y, train_lay_y, train_sit_y), axis=0)
    test_x = np.concatenate((test_jog_x, test_jump_x, test_stand_x, test_walk_x, test_lay_x, test_sit_x), axis=0)
    test_y = np.concatenate((test_jog_y, test_jump_y, test_stand_y, test_walk_y, test_lay_y, test_sit_y), axis=0)

    # Reshape the label array with one-hot coding   
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)

    if if_plot:
        # Print out proportion of dataset
        dataset_shape = list()
        dataset_shape.append(train_jog_x.shape[0])
        dataset_shape.append(train_jump_x.shape[0])
        dataset_shape.append(train_stand_x.shape[0])
        dataset_shape.append(train_walk_x.shape[0])
        dataset_shape.append(train_lay_x.shape[0])
        dataset_shape.append(train_sit_x.shape[0])

        pyplot.pie(dataset_shape,
                   explode=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                   labels=['Jogging', 'Jumping', 'Standing', 'Walking', 'Laying', 'Sitting'],
                   autopct='%.2f%%',
                   shadow=True)

        # Output figure
        pyplot.savefig('./signalFigure/TrainsetProportion.png')
        print('----  Figure saved  ----')
        pyplot.show()
        pyplot.clf()

        # Print out proportion of dataset
        dataset_shape = list()
        dataset_shape.append(train_x.shape[0])
        dataset_shape.append(test_x.shape[0])

        pyplot.pie(dataset_shape,
                   explode=[0.05, 0.05],
                   labels=['Trainset', 'Testset'],
                   autopct='%.2f%%',
                   shadow=True)

        # Output figure
        pyplot.savefig('./signalFigure/DatasetProportion.png')
        print('----  Figure saved  ----')
        pyplot.show()

    if prt_shape:
        # Print out shape of train set
        print('\nShape of train_x and train_y:')
        print(train_x.shape, train_y.shape)

        # Print out shape of the test set
        print('\nShape of test_x and test_y:')
        print(test_x.shape, test_y.shape)

    return train_x, train_y, test_x, test_y


'''
'  Method plot signal figures with given input and output
'''


def plot_figure(x, y, activity='Default', if_save=True):
    # Print out figure of the series
    pyplot.plot(x, y)
    pyplot.xlabel('Time')
    pyplot.ylabel('Values')
    # pyplot.legend(loc='upper right')

    if if_save:
        # Output figure
        pyplot.savefig('./signalFigure/Figure_' + activity + '.png')
        print('----  Figure saved  ----')

    pyplot.show()
