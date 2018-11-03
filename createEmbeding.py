import csv, os, pickle, numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))

unk_emb = np.random.uniform(-1, 1, 300)

embeddings = {}

with open(dir_path+'/../data/embedding/glove.42B.300d.tiny.txt', 'rb') as file:
    lines = file.readlines()
    for l in lines:
        elements = l.split(" ")
        embeddings[elements[0]] = [float(e) for e in elements[1:]]

with open(dir_path+'/new_dataset_train.pickle', 'rb') as dataset_file:
    dataset_train = pickle.load(dataset_file)
with open(dir_path+'/new_dataset_dev.pickle', 'rb') as dataset_file:
    dataset_dev = pickle.load(dataset_file)
with open(dir_path+'/new_dataset_test.pickle', 'rb') as dataset_file:
    dataset_test = pickle.load(dataset_file)

def embed(dataset):
    new_dataset = []
    k = 0
    for d in dataset:
        new_dataset.append([[],d[1]])
        c = 0
        j = 0
        for t in d[0]:
            if t in embeddings.keys():
                new_dataset[-1][0].append(embeddings[t])
            else:
                c += 1
                new_dataset[-1][0].append(unk_emb)
        k += c / float(len(d[0]))
    print k / float(len(dataset))
    return new_dataset

embed_train_set= embed(dataset_train)
embed_dev_set= embed(dataset_dev)
embed_test_set= embed(dataset_test)

with open(dir_path+'/embed_train_set.pickle', 'wb') as embed_train_set_file:
  pickle.dump(embed_train_set, embed_train_set_file, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_path+'/embed_dev_set.pickle', 'wb') as embed_dev_set_file:
  pickle.dump(embed_dev_set, embed_dev_set_file, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_path+'/embed_test_set.pickle', 'wb') as embed_test_set_file:
  pickle.dump(embed_test_set, embed_test_set_file, protocol=pickle.HIGHEST_PROTOCOL)

