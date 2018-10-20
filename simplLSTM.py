from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Bidirectional, Input, Dropout, BatchNormalization
import pickle, os, numpy as np
import tensorflow as tf
dir_path = os.path.dirname(os.path.realpath(__file__))

# with open(dir_path+'/new_dataset_test.pickle', 'rb') as dataset_file:
#     dataset = pickle.load(dataset_file)
#
#
# # # vocab = []
# # # max_len = 0
# # # for d in dataset:
# # #     for t in d[0]:
# # #         if t not in vocab:
# # #             vocab.append(t)
# # #     if len(d[0]) > max_len:
# # #         max_len = len(d[0])
# # #
# # # with open(dir_path+'/vocab.pickle', 'wb') as vocab_file:
# # #   pickle.dump(vocab, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)
# # # with open(dir_path+'/max_len.pickle', 'wb') as max_len_file:
# # #   pickle.dump(max_len, max_len_file, protocol=pickle.HIGHEST_PROTOCOL)
# #
with open(dir_path+'/vocab.pickle', 'rb') as vocab_file:
    vocab = pickle.load(vocab_file)
with open(dir_path+'/max_len.pickle', 'rb') as max_len_file:
    max_len = pickle.load(max_len_file)
#
# for j in xrange(len(dataset)):
#     d = dataset[j]
#     l = len(d[0])
#     for i in xrange(max_len):
#         if i >= l:
#             d[0].append(0)
#         else:
#             d[0][i] = vocab.index(d[0][i])
#
# with open(dir_path + '/index_dataset_test.pickle', 'wb') as index_dataset_file:
#     pickle.dump(dataset, index_dataset_file, protocol=pickle.HIGHEST_PROTOCOL)


with open(dir_path + '/index_dataset_train.pickle', 'rb') as dataset_file:
    dataset_train = pickle.load(dataset_file)
with open(dir_path + '/index_dataset_test.pickle', 'rb') as dataset_file:
    dataset_test = pickle.load(dataset_file)

def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

x_train = []
y_train = []
for d in dataset_train:
    x_train.append(d[0])
    y_train.append(d[1])
x_test = []
y_test = []
for d in dataset_test:
    x_test.append(d[0])
    y_test.append(d[1])

# input = Input(shape=(max_len,))
# embed = Embedding(len((vocab)),300, mask_zero=True, input_length=max_len)(input)
# lstm1 = Bidirectional(LSTM(300))(embed)
# # lstm2 = Bidirectional(LSTM(300))(lstm1)
# dense1 = Dense(300, activation='relu')(lstm1)
# dense2 = Dense(1)(dense1)
# mode = Model(inputs=input, outputs=dense2)

model = Sequential()
model.add(Embedding(len(vocab), 300, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(300, return_sequences=True)))
model.add(Bidirectional(LSTM(300, return_sequences=False)))
model.add(Dense(300, activation='relu'))
model.add(Dense(1))
model.compile(loss=huber_loss, optimizer='sgd')

print model.layers[4].output_shape
print np.asarray(y_train).shape

model.fit(np.asarray(x_train[0:100]), np.asarray(y_train[0:100]), epochs=2)

preds = model.predict(np.asarray(x_test[0:100]))

print preds[0:10]

with open(dir_path+'/predictions.pickle', 'wb') as predictions_file:
  pickle.dump(preds, predictions_file, protocol=pickle.HIGHEST_PROTOCOL)







