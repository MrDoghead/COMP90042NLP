import sys
import string
import json
import csv
import random
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.feature_selection import VarianceThreshold

def get_data_from_json(file_path):
    tweets = [] 
    labels = []
    with open(file_path,'r') as f:
        raw_data = json.load(f)
        for k,v in raw_data.items():
            text = v['text']
            text = rm_punc(text)
            tweets.append(text)
            if 'label' in v:
                labels.append(v['label'])
    if len(labels) > 0:
        return tweets,labels
    else:
        return tweets

def get_data_from_csv(file_path):
    tweets = []
    with open(file_path,'r',encoding="unicode_escape") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        rows = rows[1:]
        for each in rows:
            text = each[1].lower()
            text = rm_punc(text)
            tweets.append(text)
    labels = [0] * len(tweets)
    return tweets,labels

def get_data_from_text(file_path):
    tweets = []
    with open(file_path,'r') as f:
        for line in f:
            text = line.strip().lower()
            if text:
                text = rm_punc(text)
                tweets.append(text)
    labels = [0] * len(tweets)
    return tweets,labels

def rm_punc(text):
    for w in text:
        if w in string.punctuation:
            text = text.replace(w," ")
    return text

def prediction(fitted_model,test_x,output_file):
    pred_y = fitted_model.predict(test_x)
    res = {}
    for i in range(len(pred_y)):
        index = "test-" + str(i)
        res[index] = {"label":int(pred_y[i])}
    json.dump(res,output_file)

def view_features(fitted_dataset,vectorizer,n):
    top_n = n
    features_by_gram = defaultdict(list)
    for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
        features_by_gram[len(f.split(' '))].append((f, w))
    for gram, features in features_by_gram.items():
        top_features = sorted(features, key=lambda fitted_dataset: fitted_dataset[1], reverse=False)[:top_n]
        top_features = [f[0] for f in top_features]
        print('{}-gram top:'.format(gram), top_features)

if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)

    # import data
    print("start importing data from files...")
    dev_data,dev_labels = get_data_from_json('../project-files/dev.json')
    misinfo_train_data,misinfo_train_labels = get_data_from_json('../project-files/train.json')
    info_train_data1,info_train_labels1 = get_data_from_csv('../learn-ai-bbc/BBC News Train.csv')
    info_train_data2,info_train_labels2 = get_data_from_csv('../learn-ai-bbc/BBC News Test.csv')
    info_train_data3,info_train_labels3 = get_data_from_text('../nature_data.txt')
    info_train_data4,info_train_labels4 = get_data_from_text('../support_data.txt')
    info_train_data5,info_train_labels5 = get_data_from_json('../train_negative.json')
    info_train_data = info_train_data1 + info_train_data2 + info_train_data3 + info_train_data4 #+ info_train_data5 #+ dev_data
    info_train_labels = info_train_labels1 + info_train_labels2 + info_train_labels3 + info_train_labels4 #+ info_train_labels5 #+ dev_labels
    test_data = get_data_from_json('../project-files/test-unlabelled.json')
    assert len(misinfo_train_data) == len(misinfo_train_labels)
    assert len(info_train_data) == len(info_train_labels)
    assert len(dev_data) == len(dev_labels)
    print("completed")
    print('misinfo train size:',len(misinfo_train_data))
    print('info train size:',len(info_train_data))
    print('dev size:',len(dev_data))
    print('test size:',len(test_data))


    # make train and dev dataset
    print("start making train and dev dataset...")
    #vectorizer = DictVectorizer()
    stopwords = stopwords.words('english')
    #vectorizer = CountVectorizer(decode_error="ignore",analyzer='word',lowercase=True,stop_words=stopwords) # bow
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',ngram_range=(1, 2),decode_error="ignore",analyzer='word',lowercase=True,stop_words=stopwords)    # tfidf
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
    assert len(train_data) == len(train_labels)
    
    '''
    f = open('train.json','w')
    o = {}
    for i in range(len(train_data)):
        index = 'train-' + str(i)
        o[index] = {'text':train_data[i],'label':train_labels[i]}
    json.dump(o,f)
    sys.exit()
    '''

    train_set = vectorizer.fit_transform(train_data)
    dev_set = vectorizer.transform(dev_data)
    test_set = vectorizer.transform(test_data)
    print("completed")

    #view_features(train_data,vectorizer,10)

    # classification
    print("start classification using MNB...")
    NB_classifier = MultinomialNB(alpha=0.7,fit_prior=True,class_prior=None)
    NB_classifier.fit(train_set,train_labels)
    dev_NB_pred = NB_classifier.predict(dev_set)
    acc_NB_dev = accuracy_score(dev_labels,dev_NB_pred)
    f1_NB_dev = f1_score(dev_labels,dev_NB_pred,average='weighted')
    pre_NB_dev = precision_score(dev_labels,dev_NB_pred,average='weighted')
    rec_NB_dev = recall_score(dev_labels,dev_NB_pred,average='weighted')
    print("acc:",acc_NB_dev)
    print("F1-score:",f1_NB_dev)
    print("precision:",pre_NB_dev)
    print("recall:",rec_NB_dev)

    print("start classification using LR...")
    LR_classifier = LogisticRegression(C=6,max_iter=1000)
    LR_classifier.fit(train_set,train_labels)
    dev_LR_pred = LR_classifier.predict(dev_set)
    acc_LR_dev = accuracy_score(dev_labels,dev_LR_pred)
    f1_LR_dev = f1_score(dev_labels,dev_LR_pred,average='weighted')
    pre_LR_dev = precision_score(dev_labels,dev_LR_pred,average='weighted')
    rec_LR_dev = recall_score(dev_labels,dev_LR_pred,average='weighted')
    print("acc:",acc_LR_dev)
    print("F1-score",f1_LR_dev)
    print("precision:",pre_LR_dev)
    print("recall:",rec_LR_dev)

    print("start classification using RF...")
    RF_classifier = RandomForestClassifier(random_state=0,n_estimators=100,oob_score=True,criterion='entropy')
    RF_classifier = RF_classifier.fit(train_set,train_labels)
    dev_RF_pred = RF_classifier.predict(dev_set)
    acc_RF_dev = accuracy_score(dev_labels,dev_RF_pred)
    f1_RF_dev = f1_score(dev_labels,dev_RF_pred,average='weighted')
    pre_RF_dev = precision_score(dev_labels,dev_RF_pred,average='weighted')
    rec_RF_dev = recall_score(dev_labels,dev_RF_pred,average='weighted')
    print("acc:",acc_RF_dev)
    print("F1-score",f1_RF_dev)
    print("precision:",pre_RF_dev)
    print("recall:",rec_RF_dev)

    sys.exit()
    print("start predicting test dataset...")
    output_file = open('test-output.json','w')
    prediction(LR_classifier,test_set,output_file)
    print("completed")







