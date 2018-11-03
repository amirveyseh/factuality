# import tensorflow as tf
import statistics
from sklearn.metrics import mean_absolute_error
# from keras import backend as K
# from keras.models import Sequential, Model, load_model
# from keras.layers import Dense, Activation, Embedding, LSTM, Bidirectional, Input, Dropout, BatchNormalization, TimeDistributed, multiply, Lambda
# from tensorflow.python.keras.layers import Lambda
import pickle, os, numpy as np
# from keras_self_attention import SeqSelfAttention
dir_path = os.path.dirname(os.path.realpath(__file__))

# # with open(dir_path+'/new_dataset_train.pickle', 'rb') as dataset_file:
# #     dataset = pickle.load(dataset_file)
#
#
# # vocab = []
# # lens = []
# # for d in dataset:
# #     for t in d[0]:
# #         if t not in vocab:
# #             vocab.append(t)
# #     # if len(d[0]) > max_len:
# #     #     max_len = len(d[0])
# #     lens.append(len(d[0]))
# # lens = sorted(lens)
# # max_len = sum(lens)/len(lens)
# # print max_len
# # print statistics.stdev(lens)
# # c = 0
# # m = 0
# # for l in lens:
# #     if l <= 50:
# #         c += 1
# #     else:
# #         m += l-50
# # print c / float(len(lens))
# # print m / float(len(lens)-c)
# # exit(1)
#
# max_len = 5
# # with open(dir_path+'/vocab.pickle', 'wb') as vocab_file:
# #   pickle.dump(vocab, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)
# # with open(dir_path+'/max_len.pickle', 'wb') as max_len_file:
# #   pickle.dump(max_len, max_len_file, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# with open(dir_path+'/vocab.pickle', 'rb') as vocab_file:
#     vocab = pickle.load(vocab_file)
# # with open(dir_path+'/max_len.pickle', 'rb') as max_len_file:
# #     max_len = pickle.load(max_len_file)
# # # #
# # # # for j in xrange(len(dataset)):
# # # #     d = dataset[j]
# # # #     l = len(d[0])
# # # #     for i in xrange(max_len):
# # # #         if i >= l:
# # # #             d[0].append(0)
# # # #         else:
# # # #             try:
# # # #                 d[0][i] = vocab.index(d[0][i])
# # # #             except:
# # # #                 d[0][i] = 0
# # # #
# # # # with open(dir_path + '/index_dataset_test.pickle', 'wb') as index_dataset_file:
# # # #     pickle.dump(dataset, index_dataset_file, protocol=pickle.HIGHEST_PROTOCOL)
# # # #
# # # #
# # with open(dir_path + '/index_dataset_train.pickle', 'rb') as dataset_file:
# #     dataset_train = pickle.load(dataset_file)
# # with open(dir_path + '/index_dataset_test.pickle', 'rb') as dataset_file:
# #     dataset_test = pickle.load(dataset_file)
# #
# # with open(dir_path + '/embed_train_set.pickle', 'rb') as dataset_file:
# #     embed_train = pickle.load(dataset_file)
# # with open(dir_path + '/embed_test_set.pickle', 'rb') as dataset_file:
# #     embed_test = pickle.load(dataset_file)
# # with open(dir_path + '/embed_dev_set.pickle', 'rb') as dataset_file:
# #     embed_dev = pickle.load(dataset_file)
# #
# # def mask_zero(dataset):
# #     for d in dataset:
# #         l = len(d[0])
# #         for i in xrange(100):
# #             if i > l:
# #                 d[0].append(np.zeros(300))
# #
# # print np.asarray(embed_train[0][0]).shape
# # exit(1)
#
#
# with open(dir_path + '/embed_train_new.pickle', 'rb') as file:
#     embed_train_new = pickle.load(file)
# with open(dir_path + '/embed_train_new_label.pickle', 'rb') as file:
#     embed_train_new_label = pickle.load(file)
# with open(dir_path + '/event_words_train.pickle', 'rb') as file:
#     event_words_train = pickle.load(file)
# with open(dir_path + '/event_words_dev.pickle', 'rb') as file:
#     event_words_dev = pickle.load(file)
# with open(dir_path + '/event_words_test.pickle', 'rb') as file:
#     event_words_test = pickle.load(file)
# with open(dir_path + '/positions_train.pickle', 'rb') as file:
#     positions_train = pickle.load(file)
# with open(dir_path + '/positions_dev.pickle', 'rb') as file:
#     positions_dev = pickle.load(file)
# with open(dir_path + '/positions_test.pickle', 'rb') as file:
#     positions_test = pickle.load(file)
#
#
# def huber_loss(y_true, y_pred, clip_delta=1.0):
#   error = y_true - y_pred
#   cond  = tf.keras.backend.abs(error) < clip_delta
#
#   squared_loss = 0.5 * tf.keras.backend.square(error)
#   linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
#
#   return tf.where(cond, squared_loss, linear_loss)
# # #
# # # x_train = []
# # # y_train = []
# # # for d in dataset_train:
# # #     x_train.append(d[0][0:max_len])
# # #     y_train.append(d[1])
# # # x_test = []
# # # y_test = []
# # # for d in dataset_test:
# # #     x_test.append(d[0][0:max_len])
# # #     y_test.append(d[1])
# #
# # # #
#
# # test_slice = np.zeros((10,5,1))
# # for i in xrange(test_slice.shape[0]):
# #     test_slice[i][np.random.randint(0,5)] = [1]
# # print test_slice[0]
#
#
# def multi(inputs):
#     return tf.multiply(inputs[0], inputs[1])
# def tile(input):
#     return tf.manip.tile(input, [1, 1, 600])
# def sum(input):
#     return tf.reduce_sum(input, 1)
# def concat(inputs):
#     return tf.concat(inputs, 2)
#
# input = Input(shape=(max_len,300))
# input2 = Input(shape=(max_len,1))
# input3 = Input(shape=(max_len,))
# position_input = Embedding(2*max_len,300)(input3)
# position_embedding = Lambda(concat)([input, position_input])
# scale = Lambda(tile)(input2)
# # attention_ws = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(input)
# # attention_ts = Lambda(tile)(attention_ws)
# # weighted_input = Lambda(multi)([input, attention_ts])
# # embed = Embedding(len((vocab)),300, mask_zero=True, input_length=max_len)(input)
# lstm1 = Bidirectional(LSTM(300, return_sequences=True))(position_embedding)
# # lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True))(lstm1)
# self_attentions = SeqSelfAttention(units=300)(lstm1)
# words = Lambda(multi)([self_attentions, scale])
# verb = Lambda(sum)(words)
# dense1 = Dense(300, activation='relu')(verb)
# dense2 = Dense(1)(dense1)
# model = Model(inputs=[input, input2, input3], outputs=dense2)
# # # #
# # model = Sequential()
# # # model.add(Embedding(len(vocab), 300, input_length=max_len, mask_zero=True))
# # model.add(Bidirectional(LSTM(300, return_sequences=True), input_shape=(max_len, 300)))
# # model.add(SeqSelfAttention(attention_activation='sigmoid'))
# # model.add(Bidirectional(LSTM(300, return_sequences=False)))
# # model.add(Dense(300, activation='relu'))
# # model.add(Dense(1))
# model.compile(loss=huber_loss, optimizer='adam')
# # #
# # for layer in model.layers:
# #     print layer.name, layer.input_shape
# # print np.asarray(y_train).shape
# # #
# # # print np.asarray(x_train[0:100]).shape
# # # print np.asarray(x_train[:]).shape
# # #
# model.fit([np.asarray(embed_train_new)[0:50], event_words_train[0:50], positions_train[0:50]], np.asarray(embed_train_new_label)[0:50], epochs=2)
#
# preds = model.predict([np.asarray(embed_train_new)[0:10], event_words_train[0:10], positions_train[0:10]])
#
# print preds[0:10]

# with open(dir_path+'/predictions.pickle', 'wb') as predictions_file:
#   pickle.dump(preds, predictions_file, protocol=pickle.HIGHEST_PROTOCOL)
# with open(dir_path+'/model.pickle', 'wb') as model_file:
#   pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)



with open(dir_path + '/predictions_selfAttention.pickle', 'rb') as predictions_file:
    predictions = pickle.load(predictions_file)
# with open(dir_path + '/model.pickle', 'rb') as model_file:
#     model_weights = pickle.load(model_file)
# model.set_weights(model_weights)
with open(dir_path + '/embed_train_new_label.pickle', 'rb') as file:
    embed_train_new_label = pickle.load(file)
with open(dir_path + '/embed_test_new_label.pickle', 'rb') as file:
    embed_test_new_label = pickle.load(file)
with open(dir_path + '/embed_dev_new_label.pickle', 'rb') as file:
    embed_dev_new_label = pickle.load(file)
# with open(dir_path + '/train_predictions.pickle', 'rb') as file:
#     train_predictions = pickle.load(file)

# train_predictions = model.predict(np.asarray(y_train))

all3 = np.ones((len(embed_test_new_label)))*3
print mean_absolute_error(embed_test_new_label, all3)
print mean_absolute_error(embed_test_new_label, predictions)


# # for layer in model.layers:
# #     print layer
# inps = model.input                                           # input placeholder
# # outputs = [layer.output for layer in model.layers]          # all layer outputs
# outputs = [words, verb]
# functors = [K.function([inps[0],inps[1]], [out]) for out in outputs]    # evaluation functions
#
# # Testing
# layer_outs = [func([np.asarray(embed_train_new)[0:10], test_slice]) for func in functors]
# print layer_outs

# model_val = tf.keras.models.Model(inputs=[input, input2], outputs=[words, verb])
# my_words, my_verb = model_val.predict([np.asarray(embed_train_new)[0:1], test_slice[0:1]])
# print my_words
# print my_verb
