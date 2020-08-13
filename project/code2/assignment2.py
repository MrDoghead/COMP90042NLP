import json
import csv
import string
import random
import numpy as np
from numpy import dot
from numpy.linalg import norm
import sys
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Dropout
from keras import layers
from keras import backend as K

def get_data_from_json(file_path):
    tweets = []
    labels = []
    with open(file_path,'r') as f:
        raw_data = json.load(f)
        for k,v in raw_data.items():
            text = v['text']
            text = process(text)
            tweets.append(text)
            if 'label' in v:
                labels.append(v['label'])
    if len(labels) > 0:
        return tweets,labels
    else:
        return tweets

def process(text):
    for w in text:
        if w in string.punctuation or w in string.digits:
            text = text.replace(w," ")
    token_list = text.split()
    new_text = []
    for token in token_list:
        tmp = token.lower()
        if tmp not in stopwords:
            new_text.append(tmp)
    return ' '.join(new_text)

def get_data_from_csv(file_path):
    tweets = []
    with open(file_path,'r',encoding="unicode_escape") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        rows = rows[1:]
        for each in rows:
            text = each[1]
            text = process(text)
            tweets.append(text)
    labels = [0] * len(tweets)
    return tweets,labels

def get_data_from_text(file_path):
    tweets = []
    with open(file_path,'r') as f:
        for line in f:
            text = line.strip()
            text = process(text)
            if text:
                tweets.append(text)
    labels = [0] * len(tweets)
    return tweets,labels

def feedforward_bow(train_data,y_train,dev_data,y_dev,test_data):
    # tokenization
    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(train_data)
    x_train = tokenizer.texts_to_matrix(train_data, mode="count") #BOW
    x_dev = tokenizer.texts_to_matrix(dev_data, mode="count")
    x_test = tokenizer.texts_to_matrix(test_data, mode="count")

    vocab_size = x_train.shape[1]
    #print(x_train.shape)
    print("Vocab size =", vocab_size)
    #print(x_train[0])

    #model definition
    model = Sequential(name="feedforward-bow-input")
    model.add(layers.Dense(10, input_dim=vocab_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    #since it's a binary classification problem, we use a binary cross entropy loss here
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
    model.summary()

    # evaluting
    model.fit(x_train, y_train, epochs=20, verbose=True, validation_data=(x_dev, y_dev), batch_size=10)
    loss, accuracy, f1_score, precision, recall = model.evaluate(x_dev, y_dev, verbose=False)
    print("\naccuracy:  {:.4f}".format(accuracy))
    print("f1_score:    {:.4f}".format(f1_score))
    print("precision:   {:.4f}".format(precision))
    print("recall:  {:.4f}".format(recall))

    embedding(model,tokenizer)    # show some insights
    sys.exit()
    # predicting
    output_file = open('test-output.json','w')
    prediction(model,x_test,output_file)

def feedforward_sequence(train_data,y_train,dev_data,y_dev,test_data):
    print('feefforward-seq...')
    #tokenise the input into word sequences
    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(train_data)
    xseq_train = tokenizer.texts_to_sequences(train_data)
    xseq_dev = tokenizer.texts_to_sequences(dev_data)
    xseq_test = tokenizer.texts_to_sequences(test_data)
    vocab_size = len(word_freq) + 2
    #print(train_data[0])
    #print(xseq_train[0])
    print("Vocab size =", vocab_size)

    maxlen = 600
    xseq_train = pad_sequences(xseq_train, padding='post', maxlen=maxlen)
    xseq_dev = pad_sequences(xseq_dev, padding='post', maxlen=maxlen)
    xseq_test = pad_sequences(xseq_test, padding='post', maxlen=maxlen)
    #print(xseq_train[0])

    embedding_dim = 50

    #word order preserved with this architecture
    model = Sequential(name="feedforward-sequence-input")
    model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc',f1_m,precision_m, recall_m])
    model.summary()

    model.fit(xseq_train, y_train, epochs=20, verbose=True, validation_data=(xseq_dev, y_dev), batch_size=10)
    loss, accuracy, f1_score, precision, recall = model.evaluate(xseq_dev, y_dev, verbose=False)
    print("\naccuracy:  {:.4f}".format(accuracy))
    print("f1_score:    {:.4f}".format(f1_score))
    print("precision:   {:.4f}".format(precision))
    print("recall:  {:.4f}".format(recall))

    embedding(model,tokenizer)
    sys.exit()
    # predicting
    output_file = open('test-output.json','w')
    prediction(model,xseq_test,output_file)

def lstm(train_data,y_train,dev_data,y_dev,test_data):
    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(train_data)
    xseq_train = tokenizer.texts_to_sequences(train_data)
    xseq_dev = tokenizer.texts_to_sequences(dev_data)
    xseq_test = tokenizer.texts_to_sequences(test_data)
    maxlen = 500
    xseq_train = pad_sequences(xseq_train, padding='post', maxlen=maxlen)
    xseq_dev = pad_sequences(xseq_dev, padding='post', maxlen=maxlen)
    xseq_test = pad_sequences(xseq_test, padding='post', maxlen=maxlen)
    #print(xseq_train[0])
    embedding_dim = 50
    vocab_size = len(word_freq) + 2
    print("Vocab size =", vocab_size)

    #word order preserved with this architecture
    model = Sequential(name="lstm")
    model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc',f1_m,precision_m, recall_m])
    model.summary()

    model.fit(xseq_train, y_train, epochs=20, verbose=True, validation_data=(xseq_dev, y_dev), batch_size=10)
    loss, accuracy, f1_score, precision, recall = model.evaluate(xseq_dev, y_dev, verbose=False)
    print("\naccuracy:  {:.4f}".format(accuracy))
    print("f1_score:    {:.4f}".format(f1_score))
    print("precision:   {:.4f}".format(precision))
    print("recall:  {:.4f}".format(recall))

    embedding(model,tokenizer)

    sys.exit()
    # predicting
    output_file = open('test-output.json','w')
    prediction(model,xseq_test,output_file)

def prediction(fitted_model,x_test,output_file):
    y_pred = fitted_model.predict_classes(x_test)
    res = {}
    for i in range(len(y_pred)):
        index = "test-" + str(i)
        res[index] = {"label": int(y_pred[i])}
    json.dump(res,output_file)
    output_file.close()

def statistic(dataset):
    print('totel data:',len(dataset))
    max_length = 0
    word_freq = {}
    for text in dataset:
        words = text.split()
        if len(words) > max_length:
            max_length = len(words)
        for word in words:
            word_freq[word] = word_freq.get(word,0) + 1
    print('word_list size:',len(word_freq))
    print('max_text_length:',max_length)
    print()
    '''
    for k,v in word_freq.items():
        print(k,v)
    '''
    return word_freq,max_length

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def embedding(model,tokenizer):
    embeddings = model.get_layer(index=0).get_weights()[0] #word embeddings layer

    emb_warm = embeddings[tokenizer.word_index["warm"]]
    emb_hot = embeddings[tokenizer.word_index["hot"]]
    emb_frozen = embeddings[tokenizer.word_index["frozen"]]
    emb_cold = embeddings[tokenizer.word_index["cold"]]

    print(emb_warm)

    def cos_sim(a, b):
        return dot(a, b)/(norm(a)*norm(b))

    print("warm vs. hot =", cos_sim(emb_warm, emb_hot))
    print("frozen vs. cold =", cos_sim(emb_frozen, emb_cold))
    print("hot vs. cold =", cos_sim(emb_hot, emb_cold))
    print("warm vs. frozen =", cos_sim(emb_warm, emb_frozen))

if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    stopwords = set(stopwords.words('english'))

    # import data
    dev_data,dev_labels = get_data_from_json('../project-files/dev.json')
    misinfo_train_data,misinfo_train_labels = get_data_from_json('../project-files/train.json')
    info_train_data1,info_train_labels1 = get_data_from_csv('../learn-ai-bbc/BBC News Train.csv')
    info_train_data2,info_train_labels2 = get_data_from_csv('../learn-ai-bbc/BBC News Test.csv')
    info_train_data3,info_train_labels3 = get_data_from_text('../nature_data.txt')
    info_train_data4,info_train_labels4 = get_data_from_text('../support_data.txt')
    info_train_data5,info_train_labels5 = get_data_from_text('../train_dataset.json')
    info_train_data = info_train_data1 + info_train_data2 + info_train_data3 + info_train_data4 #+ info_train_data5 + dev_data
    info_train_labels = info_train_labels1 + info_train_labels2 + info_train_labels3 + info_train_labels4 #+ info_train_labels5 + dev_labels
    test_data = get_data_from_json('../project-files/test-unlabelled.json')
    print('misinfo train size:',len(misinfo_train_data))
    print('info train size:',len(info_train_data))
    print('dev size:',len(dev_data))
    print('test size:',len(test_data))
    print()

    # make datasets
    all_train = misinfo_train_data + info_train_data
    all_train_labels = misinfo_train_labels + info_train_labels
    zipped_train = list(zip(all_train,all_train_labels))
    random.shuffle(zipped_train)   # shuffle the data
    train_data = []
    train_labels = []
    for pair in zipped_train:
        bow,label = pair
        train_data.append(bow)
        train_labels.append(label)

    word_freq,max_text_length = statistic(train_data)   # gain some info of the training dataset

    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)
    
    # training
    #feedforward_bow(train_data,y_train,dev_data,y_dev,test_data)
    feedforward_sequence(train_data,y_train,dev_data,y_dev,test_data)
    #lstm(train_data,y_train,dev_data,y_dev,test_data)


