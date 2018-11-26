'''
A impementation of training a LSTM network to associate public bitcoin addresses, with private keys.

[  Note, inversion and non-inversion are highly suggested also wif and non-wif data, see mk-prvadd-pair.py, for examples of generating training data. This public example is only for understanding concepts, true production requires huge training sets, and various alternate representations of bitcoin address abstraction.

Reference for theory of this concept to the following paper.

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
# Bidirectional
from keras.layers import Dropout, Bidirectional, LSTM 

import numpy as np
from six.moves import range

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 50000 # was 50000, poc was 100000
#DIGITS = 3
DIGITS = 51 # but max classes * 3 is best num
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM # was LSTM, poc was GRU
#HIDDEN_SIZE = 128
HIDDEN_SIZE = 256 # > 4 times the output length
BATCH_SIZE = 128
#LAYERS = 2
LAYERS = 3
#MAXLEN = DIGITS + 1 + DIGITS
MAXLEN = 34

#chars = '0123456789+ '
import string
chars =  string.printable

ctable = CharacterTable(chars, MAXLEN)

#note here input is a column text file with private-key, and public-address, so the src is public-addr,
# and label is the private key, both in this case are in compressed 'wif' format, but all cases need to 
# considered.
 
src = 'privaddr-pair.txt'
label_file = 'privaddr-pair.txt'

# src file
with open(src,'r') as f:
    dataset = f.readlines()
    print ("Reading %s Lines %d\n" % ( src, len(dataset) ))
    contents=[]
    labels=[]
    for l in dataset:
        if len(l)>1:    # skip \n lines
            l = l.strip().split(':')
            contents.append(l[0]) # add public address field
            labels.append(l[1]) # add wif-format private-key field

# label file

'''
lines = []
#import csv
import numpy as np

with open(label_file, 'rb') as f:
    reader = csv.reader(f)
    for line in reader:
        #print row
        lines.append(line)
'''
# dont' use csv, we want comma's preserved
#with open(label_file,'r') as f:
#   labels = f.readlines()


questions = []
expected = []
i = 0
#seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    q = contents[i] # remove char > MAXLEN
    q = q[:MAXLEN]  # keep \n for learning about lines
    query = q + ' ' * (MAXLEN - len(q)) # pad line with space char

    # Answers can be of maximum size (Class) DIGITS + 1
    ans = labels[i]
    ans = ans[:len(ans)-1][:DIGITS] # remove /n and char > DIGITS
    if len(ans) == 0 :
        ans = '0' # set to zero if no classification
    ans += ' ' * (DIGITS + 1 - len(ans))
    # optional to reverse source string
    if INVERT:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
    i += 1

print('Total source lines:', len(questions))

print('Vectorization...')
X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
# don't want to shuffle the source code
'''
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
'''

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print(X_train.shape)
print(y_train.shape)

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))

# test how dropout effects
#model.add(Dropout(0.2))

# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    #model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))) # was 64


# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):  # was 200
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)

        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)

        print('---')

        # save and/or load model Keras
'''
        # serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)
'''
