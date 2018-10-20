import csv, os, pickle
dir_path = os.path.dirname(os.path.realpath(__file__))

dataset = []
sentences = {
    'dev': [],
    'train': [],
    'test': []
}

with open(dir_path+'/../data/it-happened/it-happened_eng_ud1.2_07092017.tsv', 'rb') as tsvfile:
    file = csv.reader(tsvfile, delimiter='\t')
    for r in file:
        dataset.append(r)


with open(dir_path+'/../data/ud-ewt/UD_English-EWT/en-ud-dev.conllu', 'rb') as file:
    lines = file.readlines()
    sentence = []
    for r in lines:
        if r in ['\n', '\r\n']:
            sentences['dev'].append(sentence)
            sentence = []
        else:
            ts = r.split('\t')
            sentence.append(ts[1])

with open(dir_path+'/../data/ud-ewt/UD_English-EWT/en-ud-test.conllu', 'rb') as file:
    lines = file.readlines()
    sentence = []
    for r in lines:
        if r in ['\n', '\r\n']:
            sentences['test'].append(sentence)
            sentence = []
        else:
            ts = r.split('\t')
            sentence.append(ts[1])

with open(dir_path+'/../data/ud-ewt/UD_English-EWT/en-ud-train.conllu', 'rb') as file:
    lines = file.readlines()
    sentence = []
    for r in lines:
        if r in ['\n', '\r\n']:
            sentences['train'].append(sentence)
            sentence = []
        else:
            ts = r.split('\t')
            sentence.append(ts[1])

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
    if spl != 'test':
        continue
    d.append(sentences[spl][int(ids[1])-1])
    if d[-3] != 'na':
        sign = -1
        if d[-4] == 'true' or d[-4] == 'True':
            sign = 1
        d.append(sign*int(d[-3]) * 3 / 4)
    else:
        d.append(10)


# print sentences['dev'][0:10]
# print sentences['train'][0:10]
# print sentences['test'][0:10]
# for d in dataset[0:10]:
#     print d

new_dataset = []
for d in dataset[1:-1]:
    if d[0] == 'test':
        if d[-1] != 10:
            new_dataset.append([d[-2], d[-1]])

new_data = []
for j in xrange(len(new_dataset)):
    d = new_dataset[j]
    new_data.append([[],0])
    for i in xrange(len(d[0])):
        new_data[j][0].append(str(d[0][i]))
    new_data[j][1] = int(d[1])

# print new_data[0][0][0]
# print new_data[1][0][0]
# new_data[0][0][0] = 'aaaa'
# print new_data[0][0][0]
# print new_data[1][0][0]
#
# for d in new_data[0:10]:
#     print d

new_dataset = new_data

with open(dir_path+'/new_dataset_test.pickle', 'wb') as new_dataset_file:
  pickle.dump(new_dataset, new_dataset_file, protocol=pickle.HIGHEST_PROTOCOL)

