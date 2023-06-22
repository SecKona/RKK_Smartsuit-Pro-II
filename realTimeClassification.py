import json
import socket
import threading
import tkinter as tk

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from matplotlib import pyplot

'''
    Method to process received UDP package
'''


def rec_pkt():
    # Variable initialize
    df = pd.DataFrame()
    count = 0

    # Loop to continuously receive packets
    while True:
        # Receive data and address from the socket
        data, address = udp_socket.recvfrom(16192)  # 8096

        # Decode the received data from bytes to string
        decoded_data = data.decode('utf-8')

        # Load the decoded data as JSON
        json_data = json.loads(decoded_data)

        # Print out the received JSON data
        # print(json_data)

        # Extract the data needed for prediction   
        tmp = pd.DataFrame({
            'Head.X': json_data['scene']['actors'][0]['body']['head']['position']['x'] * 100,
            'Head.Y': json_data['scene']['actors'][0]['body']['head']['position']['y'] * 100,
            'Head.Z': json_data['scene']['actors'][0]['body']['head']['position']['z'] * 100,
            'Neck.X': json_data['scene']['actors'][0]['body']['neck']['position']['x'] * 100,
            'Neck.Y': json_data['scene']['actors'][0]['body']['neck']['position']['y'] * 100,
            'Neck.Z': json_data['scene']['actors'][0]['body']['neck']['position']['z'] * 100,
            # 'Spine.X': json_data['scene']['actors'][0]['body']['spine']['position']['x'] * 100,
            # 'Spine.Y': json_data['scene']['actors'][0]['body']['spine']['position']['y'] * 100,
            # 'Spine.Z': json_data['scene']['actors'][0]['body']['spine']['position']['z'] * 100,
            'Hips.X': json_data['scene']['actors'][0]['body']['hip']['position']['x'] * 100,
            'Hips.Y': json_data['scene']['actors'][0]['body']['hip']['position']['y'] * 100,
            'Hips.Z': json_data['scene']['actors'][0]['body']['hip']['position']['z'] * 100,

            'LeftShoulder.X': json_data['scene']['actors'][0]['body']['leftShoulder']['position']['x'] * 100,
            'LeftShoulder.Y': json_data['scene']['actors'][0]['body']['leftShoulder']['position']['y'] * 100,
            'LeftShoulder.Z': json_data['scene']['actors'][0]['body']['leftShoulder']['position']['z'] * 100,
            'LeftArm.X': json_data['scene']['actors'][0]['body']['leftUpperArm']['position']['x'] * 100,
            'LeftArm.Y': json_data['scene']['actors'][0]['body']['leftUpperArm']['position']['y'] * 100,
            'LeftArm.Z': json_data['scene']['actors'][0]['body']['leftUpperArm']['position']['z'] * 100,
            'LeftForeArm.X': json_data['scene']['actors'][0]['body']['leftLowerArm']['position']['x'] * 100,
            'LeftForeArm.Y': json_data['scene']['actors'][0]['body']['leftLowerArm']['position']['y'] * 100,
            'LeftForeArm.Z': json_data['scene']['actors'][0]['body']['leftLowerArm']['position']['z'] * 100,
            'LeftHand.X': json_data['scene']['actors'][0]['body']['leftHand']['position']['x'] * 100,
            'LeftHand.Y': json_data['scene']['actors'][0]['body']['leftHand']['position']['y'] * 100,
            'LeftHand.Z': json_data['scene']['actors'][0]['body']['leftHand']['position']['z'] * 100,
            'LeftThigh.X': json_data['scene']['actors'][0]['body']['leftUpLeg']['position']['x'] * 100,
            'LeftThigh.Y': json_data['scene']['actors'][0]['body']['leftUpLeg']['position']['y'] * 100,
            'LeftThigh.Z': json_data['scene']['actors'][0]['body']['leftUpLeg']['position']['z'] * 100,
            'LeftShin.X': json_data['scene']['actors'][0]['body']['leftLeg']['position']['x'] * 100,
            'LeftShin.Y': json_data['scene']['actors'][0]['body']['leftLeg']['position']['y'] * 100,
            'LeftShin.Z': json_data['scene']['actors'][0]['body']['leftLeg']['position']['z'] * 100,
            'LeftFoot.X': json_data['scene']['actors'][0]['body']['leftFoot']['position']['x'] * 100,
            'LeftFoot.Y': json_data['scene']['actors'][0]['body']['leftFoot']['position']['y'] * 100,
            'LeftFoot.Z': json_data['scene']['actors'][0]['body']['leftFoot']['position']['z'] * 100,
            'LeftToe.X': json_data['scene']['actors'][0]['body']['leftToe']['position']['x'] * 100,
            'LeftToe.Y': json_data['scene']['actors'][0]['body']['leftToe']['position']['y'] * 100,
            'LeftToe.Z': json_data['scene']['actors'][0]['body']['leftToe']['position']['z'] * 100,
            'LeftToeEnd.X': json_data['scene']['actors'][0]['body']['leftToeEnd']['position']['x'] * 100,
            'LeftToeEnd.Y': json_data['scene']['actors'][0]['body']['leftToeEnd']['position']['y'] * 100,
            'LeftToeEnd.Z': json_data['scene']['actors'][0]['body']['leftToeEnd']['position']['z'] * 100,

            'RightShoulder.X': json_data['scene']['actors'][0]['body']['rightShoulder']['position']['x'] * 100,
            'RightShoulder.Y': json_data['scene']['actors'][0]['body']['rightShoulder']['position']['y'] * 100,
            'RightShoulder.Z': json_data['scene']['actors'][0]['body']['rightShoulder']['position']['z'] * 100,
            'RightArm.X': json_data['scene']['actors'][0]['body']['rightUpperArm']['position']['x'] * 100,
            'RightArm.Y': json_data['scene']['actors'][0]['body']['rightUpperArm']['position']['y'] * 100,
            'RightArm.Z': json_data['scene']['actors'][0]['body']['rightUpperArm']['position']['z'] * 100,
            'RightForeArm.X': json_data['scene']['actors'][0]['body']['rightLowerArm']['position']['x'] * 100,
            'RightForeArm.Y': json_data['scene']['actors'][0]['body']['rightLowerArm']['position']['y'] * 100,
            'RightForeArm.Z': json_data['scene']['actors'][0]['body']['rightLowerArm']['position']['z'] * 100,
            'RightHand.X': json_data['scene']['actors'][0]['body']['rightHand']['position']['x'] * 100,
            'RightHand.Y': json_data['scene']['actors'][0]['body']['rightHand']['position']['y'] * 100,
            'RightHand.Z': json_data['scene']['actors'][0]['body']['rightHand']['position']['z'] * 100,
            'RightThigh.X': json_data['scene']['actors'][0]['body']['rightUpLeg']['position']['x'] * 100,
            'RightThigh.Y': json_data['scene']['actors'][0]['body']['rightUpLeg']['position']['y'] * 100,
            'RightThigh.Z': json_data['scene']['actors'][0]['body']['rightUpLeg']['position']['z'] * 100,
            'RightShin.X': json_data['scene']['actors'][0]['body']['rightLeg']['position']['x'] * 100,
            'RightShin.Y': json_data['scene']['actors'][0]['body']['rightLeg']['position']['y'] * 100,
            'RightShin.Z': json_data['scene']['actors'][0]['body']['rightLeg']['position']['z'] * 100,
            'RightFoot.X': json_data['scene']['actors'][0]['body']['rightFoot']['position']['x'] * 100,
            'RightFoot.Y': json_data['scene']['actors'][0]['body']['rightFoot']['position']['y'] * 100,
            'RightFoot.Z': json_data['scene']['actors'][0]['body']['rightFoot']['position']['z'] * 100,
            'RightToe.X': json_data['scene']['actors'][0]['body']['rightToe']['position']['x'] * 100,
            'RightToe.Y': json_data['scene']['actors'][0]['body']['rightToe']['position']['y'] * 100,
            'RightToe.Z': json_data['scene']['actors'][0]['body']['rightToe']['position']['z'] * 100,
            'RightToeEnd.X': json_data['scene']['actors'][0]['body']['rightToeEnd']['position']['x'] * 100,
            'RightToeEnd.Y': json_data['scene']['actors'][0]['body']['rightToeEnd']['position']['y'] * 100,
            'RightToeEnd.Z': json_data['scene']['actors'][0]['body']['rightToeEnd']['position']['z'] * 100
        },
            index=[count])

        # Create a DataFrame from the extracted data
        df = pd.concat([df, tmp])
        count += 1

        # Output the signal window for prediction, shape = (1, window_size, features)
        # Window size = count
        if count == int(windowSize) + 1:
            # Split out height of hips as new feature column
            hips_height = df[['Hips.Y']]
            hips_height.columns = ['Hips_height']
            hips_height = hips_height.drop(0)

            # Convert into relative movement
            df = df.diff()
            df = df.drop(0)

            # Merge feature column
            df = df.join(hips_height)

            # Reshape the data to feed in model
            tmp = df.values.reshape(1, df.shape[0], df.shape[1])
            prediction = model.predict(tmp, verbose=0)
            print(prediction)
            predicted_label = np.argmax(prediction, axis=1)

            # Map one-hot code with activity type
            predicted_label = labelMapping.get(predicted_label[0], "Unknown type of activity")

            # Show predicted label
            # print(predicted_label)
            window_label1.config(text=predicted_label)

            # Reset variables
            df = pd.DataFrame()
            count = 0
            continue


'''
'  Configs
'''
modelType = 'LSTM'  # input('\nInput model type (CNN or LSTM): ')
windowSize = '32'  # input('\nInput window_size [8, 16, 32, 64, 128, 256]: ')

'''
'  Main function
'''
print("-----Initializing-----")
# Dictionary of labels
labelMapping = {0: "Jogging", 1: "Jumping", 2: "Standing", 3: "Walking", 4: "Laying", 5: "Sitting"}

# Load CNN model
model = tf.keras.models.load_model('./models/' + modelType + '_' + windowSize + '.h5')

# Create a console for showing result
window = tk.Tk()
window.title("Real-Time classifier")
window.resizable(False, False)


window_label0 = tk.Label(window, text='The performing activity:    ', font=("Courier", 16), anchor='w')
window_label0.grid(row=2, column=0, sticky='w')

window_label1 = tk.Label(window, text='......', font=("Courier", 64))
window_label1.grid(row=3, column=0)

window_label2 = tk.Label(window, text='Model type: ' + modelType, font=("Courier", 16), anchor='w')
window_label2.grid(row=0, column=0, sticky='w')

window_label3 = tk.Label(window, text='Window size: ' + windowSize, font=("Courier", 16), anchor='w')
window_label3.grid(row=1, column=0, sticky='w')

print("->Establishing connection")
# Create a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the local host and port
udp_socket.bind(('127.0.0.1', 14042))
print("->Start listening")

# Create a thread for recPkt method
thread = threading.Thread(target=rec_pkt, args=())
thread.start()
tk.mainloop()
thread.join()
