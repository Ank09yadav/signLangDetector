from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os

# map actions to labels
label_map = {label: num for num, label in enumerate(actions)}

# load and pad sequences
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path=os.path.join(DATA_PATH,action,str(sequence),f'{frame_num}.npy')
            if not os.path.exists(npy_path):
                # skip if file not found
                print(f'File not found: {npy_path}')
                continue
            res = np.load(npy_path)
            window.append(res)
        
        # Only preserve complete sequences
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
# prepare the traning and validation data
X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)       

# logs for the tensorboard
log_dir = os.path.join(os.path.dirname(__file__), 'Logs')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])
# compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tensorboard_callback])
model.summary()
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# save the model
model.save('model.h5')
# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')