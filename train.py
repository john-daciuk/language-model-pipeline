from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random
import os

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def build_model(maxlen, chars):
    model = keras.Sequential(
        [
            keras.Input(shape=(maxlen, len(chars))),
            layers.LSTM(128),
            layers.Dense(len(chars), activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model

def main():
    epochs = 40
    batch_size = 128
    input_path = os.getenv('VH_INPUTS_DIR')
    data_path = os.path.join(input_path, 'data/data.txt')
    with open(data_path, "rb") as f: 
        data = pickle.load(f)
    char_indices, indices_char, chars, maxlen = data['char_indices'], data['indices_char'], data['chars'], data['maxlen']
    arrs_path = os.path.join(input_path, 'data.npz')
    with numpy.load(arrs_path, allow_pickle=True) as f:
        x, y = f['x'], f['y']
    
    model = build_model(maxlen, chars)

    for epoch in range(epochs):
        model.fit(x, y, batch_size=batch_size, epochs=1)
        print()
        print("Generating text after epoch: %d" % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print("...Diversity:", diversity)

            generated = ""
            sentence = text[start_index : start_index + maxlen]
            print('...Generating with seed: "' + sentence + '"')

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.0
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                sentence = sentence[1:] + next_char
                generated += next_char

            print("...Generated: ", generated)
            print()

            OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './outputs')
            model.save(os.path.join(OUTPUTS_DIR, 'model.h5'), model)

    
        
if __name__ == '__main__':
  main()
