from tensorflow import keras
import io, os
import numpy as np


def main():
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
    maxlen = 40
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
    np.savez(os.path.join(OUTPUTS_DIR, 'data'), x=x, y=y)


if __name__ == '__main__':
    main()
