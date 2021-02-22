from tensorflow import keras
import io, os
import numpy as np
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxlen', type=int, default=40)
    return parser.parse_args()

def main():
    args = parse_args()
    path = keras.utils.get_file(
        "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
    )
    with io.open(path, encoding="utf-8") as f:
        text = f.read().lower()
    text = text.replace("\n", " ")  # We remove newlines chars for nicer display
    print("Corpus length:", len(text))

    chars = sorted(list(set(text)))
    print("Total chars:", len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = args.maxlen
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i : i + maxlen])
        next_chars.append(text[i + maxlen])
    print("Number of sequences:", len(sentences))

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    
    OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './outputs')
    np.savez(os.path.join(OUTPUTS_DIR, 'arrs'), x=x, y=y)
    data = {'char_indices':char_indices, 'indices_char':indices_char, 'chars':chars, 'maxlen':maxlen, 'text':text}
    with open(os.path.join(OUTPUTS_DIR, 'data'), "wb") as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
