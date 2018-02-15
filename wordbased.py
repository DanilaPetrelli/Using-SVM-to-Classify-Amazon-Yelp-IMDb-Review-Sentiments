from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
import random
import nltk

# read data from file and put it in a correct format

text=open("alldata_labelled.txt").read()

lines = open('alldata_labelled.txt').readlines()
random.shuffle(lines)
open('alldata_shuffled.txt', 'w').writelines(lines)

def process_line(line):
    text, label = line.split('\t', 1)
    text = text.strip().lower()
    # if maxlen:
    #     text = text[:maxlen]
    return text, label

# represent data: features selection
# update the dict freqs with the count of every word in the sentence and then return 
# a frq distribution with the counter

def represent(text):
    text = nltk.word_tokenize(text)
    all_freqs = Counter()
    freqs = Counter()
    for word in text: 
        freqs[word] += 1
        all_freqs.update(freqs)
          
    return all_freqs

new_document = open("alldata_shuffled.txt").read()

if __name__=='__main__':
   
    length = len(new_document.split())
    
    for maxlen in [length]:  

        X_all = []
        Y_all = []
        with open('alldata_shuffled.txt',encoding="utf-8") as f:
         
            for l in f:
                t, lbl = process_line(l)
                #print (t, lbl)

                X_all.append(represent(t))
                #print (X_all)
                Y_all.append(lbl[:-1])
                
                #print(Y_all)
                             
        X_train = X_all 
        Y_train = Y_all 
        train_sents = X_all[:int(len(X_all)*0.8)]
        test_sents = X_all[int(len(X_all)*0.8):]
        train_labels = Y_all[:int(len(X_all)*0.8)]
        test_labels = Y_all[int(len(X_all)*0.8):]

		#clf = svm.LinearSVC()
        clf = Pipeline([("count_vectorizer", DictVectorizer()),
						("nrm", Normalizer(norm='l2', copy= True)),
						("tfidf", TfidfTransformer()),
						("clf", LinearSVC())])
        clf.fit(train_sents, train_labels)

        predict= clf.predict(test_sents)

        print("-"*30)
        print("WORD-BASED")
        print("accuracy:", "{0:.2f}%".format(accuracy_score(test_labels, predict, normalize= True) * 100))
        print("confusion matrix:\n", confusion_matrix(test_labels, predict, labels=["POS", "NEG"]))
        print("-"*30)

# #test sentences examples
#         correct = []
#         wrong = []
#         for i in zip(test_sents,test_labels, predict):
#             if i[1]== i[2]:
#                 correct.append((i[0], i[1], i[2]))
#             else:
#                 wrong.append((i[0], i[1], i[2]))
#         #print("Correct", correct)
#         print("Incorrect", wrong)