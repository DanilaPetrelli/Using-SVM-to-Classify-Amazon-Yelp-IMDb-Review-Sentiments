from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
import nltk

# put data in a correct format
def process_line(line):
	text, label = line.split('\t', 1)
	text = text.strip()
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
		  
# read data from file

amazon = open("amazon_cells_labelled.txt").read()
imdb =  open("imdb_labelled.txt").read()
yelp = open("yelp_labelled.txt").read()

if __name__=='__main__':

	print("-"*30)
	print("WORD-BASED")

	length = len(amazon.split())
	
	for maxlen in [length]:  

		X_all = []
		Y_all = []
		with open('amazon_cells_labelled.txt',encoding="utf-8") as f:
		 
			for l in f:
				t, lbl = process_line(l)
				#print (t, lbl)

				X_all.append(represent(t))
				#print (X_all)
				Y_all.append(lbl[:-1])
				
				#print(X_all)
							 
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
		print("Amazon:")
		print("accuracy:", "{0:.2f}%".format(accuracy_score(test_labels, predict, normalize= True) * 100))
		print("confusion matrix:\n", confusion_matrix(test_labels, predict, labels=["1", "0"]))

	length = len(imdb.split())
	
	for maxlen in [length]:  

		X_all = []
		Y_all = []
		with open('imdb_labelled.txt',encoding="utf-8") as f:
		 
			for l in f:
				t, lbl = process_line(l)
				#print (t, lbl)

				X_all.append(represent(t))
				#print (X_all)
				Y_all.append(lbl[:-1])
				
				#print(X_all)
							 
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
		print("IMDb:")
		print("accuracy:", "{0:.2f}%".format(accuracy_score(test_labels, predict, normalize= True) * 100))
		print("confusion matrix:\n", confusion_matrix(test_labels, predict, labels=["1", "0"]))

	length = len(yelp.split())
	
	for maxlen in [length]:  

		X_all = []
		Y_all = []
		with open('yelp_labelled.txt',encoding="utf-8") as f:
		 
			for l in f:
				t, lbl = process_line(l)
				#print (t, lbl)

				X_all.append(represent(t))
				#print (X_all)
				Y_all.append(lbl[:-1])
				
				#print(X_all)
							 
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
		print("Yelp:")
		print("accuracy:", "{0:.2f}%".format(accuracy_score(test_labels, predict, normalize= True) * 100))
		print("confusion matrix:\n", confusion_matrix(test_labels, predict, labels=["1", "0"]))

# #test sentences examples
# 		correct = []
# 		wrong = []
# 		for i in zip(test_sents,test_labels, predict):
# 			if i[1]== i[2]:
# 				correct.append((i[0], i[1], i[2]))
# 			else:
# 				wrong.append((i[0], i[1], i[2]))
# 		#print("Correct", correct)
# 		print("Incorrect", wrong)