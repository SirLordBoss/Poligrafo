#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize

#string_a = "Trump claims that the U.S. Covid-19 mortality rate was the world's best"

#string_b = "Data reveal that the Covid-19 mortality rate in the U.S. is not so good."

#a_list = word_tokenize(string_a)
#b_list = word_tokenize(string_b)

#stopwords = stopwords.words('english')

#l1 = []
#l2 = []

#a_set = {w for w in a_list if not w in stopwords}
#b_set = {w for w in b_list if not w in stopwords}

#rvector = a_set.union(b_set)

#for w in rvector:
#    if w in a_set:
#        l1.append(1)
#    else:
#        l1.append(0)
#    if w in b_set:
#        l2.append(1)
#    else:
#        l2.append(0)

#c = 0

#for i in range(len(rvector)):
#    c += l1[i]*l2[i]

#cosine = c / float((sum(l1)*sum(l2))**0.5)
#print("similarity: ", cosine)

from sent2vec.vectorizer import Vectorizer
from scipy import spatial

sentences = [
    "The boy quickly ran across the finish line, seizing yet another victory.",
    "The child quickly ran across the finish line, seizing yet another win."
]

#pre-processing

vectorizer = Vectorizer()
vectorizer.bert(sentences)
vectors_bert = vectorizer.vectors

dist_1 = spatial.distance.cosine(vectors_bert[0], vectors_bert[1])

print('dist_1: {0}'.format(dist_1))

#TO-DO LIST:
#testar o dataset fornecido com o primeiro metodo
#explorar embeddings
