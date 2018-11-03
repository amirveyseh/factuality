import csv, os, pickle, numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))

dataset = []
sentences = {
    'dev': [],
    'train': [],
    'test': []
}
trees = {
    'dev': [],
    'train': [],
    'test': []
}
levels = {
    'dev': [],
    'train': [],
    'test': []
}
max_len = 5

def read_sentence(datasetname, file):
    lines = file.readlines()
    sentence = []
    tree = {}
    relations = []
    j = 0
    for r in lines:
        j += 1
        if r in ['\n', '\r\n']:
            sentences[datasetname].append(sentence)
            sentence = []
            for p in relations:
                if p[1] not in tree.keys():
                    tree[p[1]] = [p[0]]
                else:
                    tree[p[1]].append(p[0])
            trees[datasetname].append(tree)
            tree = {}
            relations = []
        else:
            ts = r.split('\t')
            sentence.append(ts[2])
            relations.append([int(ts[0])-1,int(ts[6])-1])

def get_levels(trees, datasetname):
    i = 0
    for tree in trees[datasetname]:
        levels[datasetname].append([])
        nodes = set().union([n for v in tree.values() for n in v])
        for n in nodes:
            l = get_level(tree, n)
            levels[datasetname][i].append([n, l])
        i += 1


def get_level(tree, n, level= 0):
    for p in tree.keys():
        if n in tree[p]:
            return get_level(tree, p, level+1)
    return level


def create_dataset(new_dataset, d, levels, tree):
    if d[-1] != 10:
        new_dataset.append([d[-2], d[-1]])
        # sorted_levels = sorted(levels, key=lambda t: -1*t[1])
        # sentence = []
        # for t in sorted_levels:
        #     sentence.append(d[-3][t[0]])
        # new_dataset.append(sentence)
        # new_dataset.append(tree)

def correct_dataset(dataset):
    new_data = []
    for j in xrange(len(dataset)):
        d = dataset[j]
        new_data.append([[], 0])
        for i in xrange(len(d[0])):
            new_data[j][0].append(str(d[0][i]))
        new_data[j][1] = int(d[1])
    return new_data




with open(dir_path+'/../data/it-happened/it-happened_eng_ud1.2_07092017.tsv', 'rb') as tsvfile:
    file = csv.reader(tsvfile, delimiter='\t')
    for r in file:
        dataset.append(r)

with open(dir_path+'/../data/ud-ewt/UD_English-EWT/en-ud-dev.conllu', 'rb') as file:
    read_sentence('dev', file)

with open(dir_path+'/../data/ud-ewt/UD_English-EWT/en-ud-test.conllu', 'rb') as file:
    read_sentence('test', file)

with open(dir_path+'/../data/ud-ewt/UD_English-EWT/en-ud-train.conllu', 'rb') as file:
    read_sentence('train', file)

# print len(sentences['dev']), len(trees['dev'])
# print len(sentences['train']), len(trees['train'])
# print len(sentences['test']), len(trees['test'])


for i in xrange(1,len(dataset)):
    d = dataset[i]
    ids = d[3].split(' ')
    spl = None
    if 'dev' in ids[0]:
        spl = 'dev'
    elif 'test' in ids[0]:
        spl = 'test'
    elif 'train' in ids[0]:
        spl = 'train'
    # if spl != 'test':
    #     continue
    d.append(sentences[spl][int(ids[1])-1])
    if d[-3] != 'na':
        sign = -1
        if d[-4] == 'true' or d[-4] == 'True':
            sign = 1
        d.append(sign*int(d[-3]) * 3 / 4)
    else:
        d.append(10)
    # d.append(trees[spl][int(ids[1])-1])



# print sentences['dev'][0:10]
# print sentences['train'][0:10]
# print sentences['test'][0:10]
# for d in dataset[0:10]:
#     print d

# get_levels(trees, 'dev')
# get_levels(trees, 'train')
# get_levels(trees, 'test')

new_dataset_train = []
new_dataset_test = []
new_dataset_dev = []
i = 0
for d in dataset[1:-1]:
    if d[0] == 'test':
        # create_dataset(new_dataset_test, d, levels['test'][i], trees['test'][i])
        create_dataset(new_dataset_test, d, None, None)
    if d[0] == 'train':
        # create_dataset(new_dataset_train, d, levels['train'][i], trees['train'][i])
        create_dataset(new_dataset_train, d, None, None)
    if d[0] == 'dev':
        # create_dataset(new_dataset_dev, d, levels['dev'][i], trees['dev'][i])
        create_dataset(new_dataset_dev, d, None, None)
    i += 1

# for d in new_dataset_train[0:10]:
#     print d

new_data_train = correct_dataset(new_dataset_train)
new_data_dev = correct_dataset(new_dataset_dev)
new_data_test = correct_dataset(new_dataset_test)

# print new_data[0][0][0]
# print new_data[1][0][0]
# new_data[0][0][0] = 'aaaa'
# print new_data[0][0][0]
# print new_data[1][0][0]
#
# for d in new_data[0:10]:
#     print d

new_dataset_train = new_data_train
new_dataset_dev = new_data_dev
new_dataset_test = new_data_test

event_words_train = np.zeros((len(new_dataset_train), max_len, 1))
event_words_dev = np.zeros((len(new_dataset_dev), max_len, 1))
event_words_test = np.zeros((len(new_dataset_test), max_len, 1))
j_train = 0
j_test = 0
j_dev = 0
for d in dataset[1:-1]:
        if d[-1] != 10:
            ind = int(d[4])-1
            if ind >= max_len:
                ind = max_len-1
            if d[0] == 'train':
                event_words_train[j_train][ind] = [1]
                j_train += 1
            if d[0] == 'dev':
                event_words_dev[j_dev][ind] = [1]
                j_dev += 1
            if d[0] == 'test':
                event_words_test[j_test][ind] = [1]
                j_test += 1

positions_train = np.zeros((len(new_dataset_train), max_len))
positions_dev = np.zeros((len(new_dataset_dev), max_len))
positions_test = np.zeros((len(new_dataset_test), max_len))
j_train = 0
j_test = 0
j_dev = 0
for d in dataset[1:-1]:
        if d[-1] != 10:
            ind = int(d[4])-1
            if ind >= max_len:
                ind = max_len-1
            if d[0] == 'train':
                for i in range(max_len):
                    positions_train[j_train][i] = max_len + i - ind
                j_train += 1
            if d[0] == 'dev':
                for i in range(max_len):
                    positions_dev[j_dev][i] = max_len + i - ind
                j_dev += 1
            if d[0] == 'test':
                for i in range(max_len):
                    positions_test[j_test][i] = max_len + i - ind
                j_test += 1

#
# for i in range(10):
#     print event_words_train[i].flatten()
#     print positions_train[i].flatten()


# with open(dir_path+'/new_dataset_train.pickle', 'wb') as new_dataset_file:
#   pickle.dump(new_dataset_train, new_dataset_file, protocol=pickle.HIGHEST_PROTOCOL)
# with open(dir_path+'/new_dataset_test.pickle', 'wb') as new_dataset_file:
#   pickle.dump(new_dataset_test, new_dataset_file, protocol=pickle.HIGHEST_PROTOCOL)
# with open(dir_path+'/new_dataset_dev.pickle', 'wb') as new_dataset_file:
#   pickle.dump(new_dataset_dev, new_dataset_file, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_path+'/event_words_train.pickle', 'wb') as event_words_file:
  pickle.dump(event_words_train, event_words_file, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_path+'/event_words_dev.pickle', 'wb') as event_words_file:
  pickle.dump(event_words_dev, event_words_file, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_path+'/event_words_test.pickle', 'wb') as event_words_file:
  pickle.dump(event_words_test, event_words_file, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_path+'/positions_train.pickle', 'wb') as positions_train_file:
  pickle.dump(positions_train, positions_train_file, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_path+'/positions_dev.pickle', 'wb') as positions_dev_file:
  pickle.dump(positions_dev, positions_dev_file, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_path+'/positions_test.pickle', 'wb') as positions_test_file:
  pickle.dump(positions_test, positions_test_file, protocol=pickle.HIGHEST_PROTOCOL)




