# RKK_Smartsuit-Pro-II

Project 'RKK_Smartsuit-Pro-II' includes code used in the thesis. 

## Author

Jierui Li, aka 'SecKona' https://github.com/SecKona

## Environment & devices

You can download this project and import it into Pycharm IDE, settings maybe automaticly initialized
* Devices: ROKOKO smart suit pro II, WIFI router and computer installed with following software.
* Software: ROKOKO studio release-2.4.1.63 (aaf09ad), Pycharm community edition 2023.1
* Libraries: python 3.11.0, Tensorflow 2.12.0, numpy, pandas, matplotlib

## Files in the folder

The file directory could be divided as follow:

### RKK_Records & csv_original

This folder includes .bvh files output from RKK studio and corresponding .csv files.
* 'bvh_convert.bat' a bat file to automaticly convert 6 classes bvh files into csv files
* How to output recorded bvh files: use module of 'Export' in RKK studio with default settings, note that the data head may be different for advanced version.

### models

This folder includes saved models

### 'loadDataset.py'

This python source code includes funtion used for loading training data from folder 'csv_original' into specified format, and performing 'sliding window' approach.

### 'model_***.py'

This kind of python source codes include the structure of the model used in training process.

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
* Prerequisites: connect smart suit with RKK studio, and use 'livestream' module to set stream with udp address/port (This function should upgrade account plans, so if it is possible, create a program to analyze udp packet sent from smart suit directly would be better (Actually I failed to analyze the packet because I don't know the data head of it  :-(     ))

Note: this kind of approach perform classification with an interval according to the window size, maybe it's possible to realize 'real real-time' for every sample frame using a buffer-like container. (i.e.,  dataframe -> [queue] -> model => activity, add a new frame into the queue , meanwhile delete the earliest one)
