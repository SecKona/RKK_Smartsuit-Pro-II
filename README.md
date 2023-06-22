# RKK_Smartsuit-Pro-II

Project 'RKK_Smartsuit-Pro-II' includes code used in the thesis "Real-Time Human Activity Recognition using a Full-Body tracking suit". 

## Author

Jierui Li, aka 'SecKona' https://github.com/SecKona

## Environment & devices

You can download this project and import it into Pycharm IDE, settings maybe automaticly initialized
* Devices: ROKOKO smart suit pro II, WIFI router and computerinstalled with following software.
* Operating system: Windows 10 release 22H2
* Software: ROKOKO studio release-2.4.1.63 (aaf09ad), Pycharm community edition 2023.1
* Libraries: python 3.11.0, Tensorflow 2.12.0, numpy, pandas, matplotlib, pip, graphviz for model structure drawings
* More tutorials: Tensorflow: https://www.tensorflow.org/; Pycharm: https://www.jetbrains.com/pycharm/; ROKOKO: https://www.rokoko.com/

## Files in the folder

The file directory could be divided as follow:

### RKK_Records (Original data) & csv_original (Annotated)

This folder includes .bvh files output from RKK studio and corresponding .csv files.
* 'bvh_convert.bat' a bat file to automaticly convert specified 6 classes bvh files into csv files (cmd: bvh-converter ****.bvh)
* How to output recorded bvh files: use module of 'Export' in RKK studio with default settings, note that the data head may be different for advanced version.
* Before using converted csv files, it should be handcraft annotated (In EXCEL: add a column 'Activity' and mark the type of activity)

### models

This folder includes saved models

### 'loadDataset.py'

This python source code includes funtion used for loading training data from folder 'csv_original' into specified format, and performing 'sliding window' approach, reshape them into required shape to feed in models.

### 'model_***.py'

This kind of python source codes include the structure of the model used in training process.

Note: the model params could be automaticly fine-tuned by using tensorflow 'Keras Tuner' API

### 'plotCMFigure.py'

This python source code will load trained models in folder 'models' and load testset for classification, then create confusion matrices into folder 'confusionMatrices' 

### 'plotSignalFigure.py'

This python source code will load training data, plot and save figures into folder 'signalFigure'

### '***Test.py'

These python source codes will load testset and use them for testing (test round = 10), meanwhile record the time consumption/accuracy and save box plots into folder 'paramTestFigure'

### 'trainModel.py' and 'trainModelAuto.py'

These codes perform model training approach and save loss curve/accuracy curve with respect to training epochs into folder 'history'
* 'trainModel.py' function to train single model and decide if save it
* 'trainModelAuto.py' function to train models with given window sizes altogether

### 'realTimeClassification.py'

This source code includes function used for prototype of real-time classifier
* Prerequisites: connect smart suit with RKK studio, and use 'livestreaming' module to set stream with udp address/port corresponding to the configuration in this source code. (This function 'livestreaming' should upgrade RKK account plans, so if it is possible, create a program to analyze udp packet (i.e., raw IMU data) sent from smart suit directly would be better (Actually I failed to analyze the packet because I don't know the data head of it  :-(     ))

Note: this kind of approach perform classification with an interval according to the window size, maybe it's possible to realize 'real real-time' for every sample frame using a buffer-like container. (i.e.,  dataframe -> [queue] -> model => activity, add a new frame into the queue , meanwhile delete the earliest one)
