import time
import numpy
import nltk

from paraphrases import msrpcx

from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

import numpy as np

import matplotlib.pyplot as plt

class Embeddings:
    def __init__(self, dictEmbeddings: set):
        self.dictEmbeddings = dictEmbeddings

    @staticmethod
    def cosim(u, v) -> float:
        return 1.0 - cosine(u, v)

    @staticmethod
    def line2WVector(line):
        try:
            vs = line.split()
            v = [float(x) for x in vs[1:]]
            return vs[0], v
        except:
            return None, None

    @staticmethod
    def loadEmbeddingsFromFile(source: str):
        dictEmbeddings = {}
        try:
            file = open(source, encoding='utf-8')
            for line in file:
                word, lista = Embeddings.line2WVector(line.strip())
                if (word is not None) and word.isalpha():
                    dictEmbeddings[word.lower()] = numpy.array(lista)
                    #if len(dictEmbeddings) > 10000: break
            file.close()
            return Embeddings(dictEmbeddings)
        except:
            return None

    def length(self) -> int:
        return len(self.dictEmbeddings)

    def getDimension(self):
        n = len(next(iter(self.dictEmbeddings.values())))
        return n

    def getVector(self, word: str):
        if word in self.dictEmbeddings.keys():
            return self.dictEmbeddings[word]
        else:
            return None

    def getVectorSum(self, s: str):
        tokens = nltk.word_tokenize(s)
        vsoma = numpy.array(self.getDimension() * [0])
        for t in tokens:
            vt = self.getVector(t)
            if vt is not None:
                vsoma = vsoma + vt
        return vsoma

    def getVectorAvg(self, s: str):
        v_average = 0
        ##meter a logica
        return v_average

    def simSumVectCos(self, sa: str, sb: str) -> float:
        a = self.getVectorSum(sa)
        b = self.getVectorSum(sb)
        return self.cosim(a, b)

    def simBestConnect(self, sa: str, sb: str):
        ta = nltk.word_tokenize(sa)
        na = len(ta)
        tb = nltk.word_tokenize(sb)
        nb = len(tb)
        if na < nb:
            ta, tb = tb, ta
            na, nb = nb, na

        A = numpy.ones((nb, na))
        for i in range(nb):
            vi = self.getVector(tb[i])
            if vi is not None:
                for j in range(na):
                    vj = self.getVector(ta[j])
                    if vj is not None:
                        A[i][j] = 1.0 - self.cosim(vi, vj)  # ==> Cost
        soma = 0.0
        b_idx, a_idx = linear_sum_assignment(A)
        for i in b_idx:
            j = a_idx[i]
            if A[i][j] < 0.3:
                soma += 1.0-A[i][j]
            #print("%.7f  %s  %s" % (A[i][j], tb[i], ta[j]))

        return soma/nb


if __name__ == '__main__':
    t0 = time.time()
    print('LOADING WORD EMBEDDINGS .....')
    emb = Embeddings.loadEmbeddingsFromFile(
        "D:\\Users\\ruica\\Projeto_FKNews\\GNewsW2Vsingle.txt")
    dt = time.time() - t0
    if emb is None:
        print("ERROR LOADING SOURCE FILE!")
    else:
        print('OK:.... %d    dt: %6.2f s' % (emb.length(), dt))

    counter = 0

    similarity_vec = []

    true_results = 0
    false_results = 0

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    precision = 0
    recall = 0
    specificity = 0
    f_measure = 0

    best_threshold = 0
    best_precision = 0
    best_recall = 0
    best_specificity = 0
    best_f_measure = 0

    x_axis = []
    y_axis = []

    for paraphrase, string_a, string_b in msrpcx:
        #print(paraphrase)
        #print(string_a)
        #print(embed([string_a])[0])
        #print(string_b)
        #print(embed([string_b])[0])

        similarity_vec.append([emb.cosim(emb.getVectorSum(string_a), emb.getVectorSum(string_b)), paraphrase])
        counter += 1
        print("similarity: " + str(counter) + ", paraphrase: " + str(paraphrase))

    with open("results_Gnews_embeddings.txt", "w") as results_file:
        for threshold in np.arange(0.1, 0.91, 0.01):
            counter = 0
            print("Threshold: " + str(threshold))

            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0

            precision = 0
            recall = 0
            specificity = 0
            f_measure = 0

            x_axis.append(threshold)

            for similarity, paraphrase in similarity_vec:
                #print(counter)
                if(similarity > threshold):
                    #print("detected paraphrase")
                    if(paraphrase == 1):
                        #print("true result")
                        true_positives += 1
                    else:
                        #print("false result")
                        false_positives += 1
                else:
                    #print("did not detect paraphrase")
                    if(paraphrase == 1):
                        #print("false result")
                        false_negatives += 1
                    else:
                        #print("true result")
                        true_negatives += 1
                #print()
                counter += 1

            print("True positives: " + str(true_positives))
            print("true negatives: " + str(true_negatives))
            print("false positives: " + str(false_positives))
            print("false negatives: " + str(false_negatives))

            precision = true_positives/(true_positives + false_positives)

            recall = true_positives/(true_positives + false_negatives)

            specificity = true_negatives/(true_negatives + false_positives)

            f_measure = 2 * ((precision * recall)/(precision + recall))

            y_axis.append(f_measure)

            if(f_measure > best_f_measure):
                best_threshold = threshold
                best_f_measure = f_measure
                best_precision = precision
                best_recall = recall
                best_specificity = specificity

            results_file.write("threshold: " + str(threshold) + "\n")

            results_file.write("Precision: " + str(((true_positives) /
                                                    (true_positives + false_positives))*100) + "%\n")
            #print("Precision: " + str(((true_positives) / (true_positives + false_positives))*100) + "%")

            results_file.write("Recall: " + str(((true_positives) /
                                                (true_positives + false_negatives))*100) + "%\n")
            #print("Recall: " + str(((true_positives) / (true_positives + false_negatives))*100) + "%")

            results_file.write("Specificity: " + str(((true_negatives) /
                                                    (true_negatives + false_positives))*100) + "%\n")
            #print("Specificity: " + str(((true_negatives) / (true_negatives + false_positives))*100) + "%")

            results_file.write("F-measure: " + str(f_measure) + "\n")
            #print("F-measure: " + str(f_measure))

            results_file.write("\n")

        results_file.write("Best threshold: " + str(best_threshold) + "\n")
        results_file.write("Best f-measure: " + str(best_f_measure) + "\n")
        results_file.write("Best precision: " + str(best_precision) + "\n")
        results_file.write("Best recall: " + str(best_recall) + "\n")
        results_file.write("Best specificity: " + str(best_specificity) + "\n")

        plt.plot(x_axis, y_axis, label="f-measure")

        plt.xlabel('threshold')
        # naming the y axis
        plt.ylabel('F-measure')

        # giving a title to my graph
        plt.title('Gnews embeddings')

        # function to show the plot, after saving it
        plt.savefig('gnews_graph.png')
        plt.show()
