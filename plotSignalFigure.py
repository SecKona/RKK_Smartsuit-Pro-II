import loadDataset
import pandas as pd
from matplotlib import pyplot
'''
'  Config
'''
activity = ['Jogging', 'Jumping', 'Standing', 'Walking', 'Laying', 'Sitting']

'''
'  Main function
'''
# Print out figure of the series
for act in activity:
    print('Plotting activity: ' + act)

    # Load CSV file
    tmp = pd.read_csv('./csv_original/' + act + '_worldpos.csv', header=0)
    tmp.pop('Activity')
    tmstmp = tmp.pop('Time')

    # Spilt out specified columns
    tmpcol = tmp.loc[:, tmp.columns.str.endswith('.X')]

    # Plot figures
    pyplot.clf()
    pyplot.plot(tmstmp, tmpcol)
    pyplot.title(act + '.X')
    pyplot.xlabel('Time')
    pyplot.ylabel('Values')
    pyplot.savefig('./signalFigure/Absolute/' + act + '_X.png')

    # Spilt out specified columns
    tmpcol = tmp.loc[:, tmp.columns.str.endswith('.Y')]

    # Plot figures
    pyplot.clf()
    pyplot.plot(tmstmp, tmpcol)
    pyplot.title(act + '.Y')
    pyplot.xlabel('Time')
    pyplot.ylabel('Values')
    pyplot.savefig('./signalFigure/Absolute/' + act + '_Y.png')

    # Spilt out specified columns
    tmpcol = tmp.loc[:, tmp.columns.str.endswith('.Z')]

    # Plot figures
    pyplot.clf()
    pyplot.plot(tmstmp, tmpcol)
    pyplot.title(act + '.Z')
    pyplot.xlabel('Time')
    pyplot.ylabel('Values')
    pyplot.savefig('./signalFigure/Absolute/' + act + '_Z.png')

print('---- Plot finished ----')

# Print out figure of the series
for act in activity:
    print('Activity: ' + act)

    # Load CSV file
    tmp = pd.read_csv('./csv_original/' + act + '_worldpos.csv', header=0)
    tmp.pop('Activity')
    tmstmp = tmp.pop('Time')
    tmstmp = tmstmp.drop(0)

    # Convert into relative movement
    tmp = tmp.diff()
    tmp = tmp.drop(0)

    # Spilt out specified columns
    tmpcol = tmp.loc[:, tmp.columns.str.endswith('.X')]

    # Plot figures
    pyplot.clf()
    pyplot.plot(tmstmp, tmpcol)
    pyplot.title(act + '.X')
    pyplot.xlabel('Time')
    pyplot.ylabel('Values')
    pyplot.savefig('./signalFigure/Relative/' + act + '_X.png')

    # Spilt out specified columns
    tmpcol = tmp.loc[:, tmp.columns.str.endswith('.Y')]

    # Plot figures
    pyplot.clf()
    pyplot.plot(tmstmp, tmpcol)
    pyplot.title(act + '.Y')
    pyplot.xlabel('Time')
    pyplot.ylabel('Values')
    pyplot.savefig('./signalFigure/Relative/' + act + '_Y.png')

    # Spilt out specified columns
    tmpcol = tmp.loc[:, tmp.columns.str.endswith('.Z')]

    # Plot figures
    pyplot.clf()
    pyplot.plot(tmstmp, tmpcol)
    pyplot.title(act + '.Z')
    pyplot.xlabel('Time')
    pyplot.ylabel('Values')
    pyplot.savefig('./signalFigure/Relative/' + act + '_Z.png')

print('---- Plot finished ----')
