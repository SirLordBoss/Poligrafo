import time
import numpy
import nltk

from paraphrases import msrpcx

from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

import numpy as np

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

    true_results = []
    false_results = []

    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    precision = 0
    recall = 0
    specificity = 0
    f_measure = 0

    with open("results_Gnews_embeddings.txt", "w") as results_file:
        for threshold in np.arange(0.1, 0.91, 0.01):
            for paraphrase, string_a, string_b in msrpcx:
                print(counter)
                #print(paraphrase)
                #print(string_a)
                #print(embed([string_a])[0])
                #print(string_b)
                #print(embed([string_b])[0])

                similarity = emb.cosim(emb.getVectorSum(string_a), emb.getVectorSum(string_b))

                print("similarity: ", similarity)

                #ciclo for para percorrer diferentes thresholds: 0.1 -> 0.9 (step=0.01) 
                #guardar em termo de f-measure
                #fazer função para isto
                if(similarity > threshold):
                    #print("detected paraphrase")
                    if(paraphrase == 1):
                        #print("true result")
                        true_results += 1
                        true_positives += 1
                    else:
                        #print("false result")
                        false_results += 1
                        false_positives += 1
                else:
                    #print("did not detect paraphrase")
                    if(paraphrase == 1):
                        #print("false result")
                        false_results += 1
                        false_negatives += 1
                    else:
                        #print("true result")
                        true_results += 1
                        true_negatives += 1

                #print()
                counter += 1
            
            precision = true_positives/(true_positives + false_positives)

            recall = true_positives/(true_positives + false_negatives)

            specificity = true_negatives/(true_negatives + false_positives)

            f_measure = 2 * ((precision * recall)/(precision + recall))

            #print("Final analysis")

            results_file.write("Threshold: " + str(threshold) + "\n")
            #print("True results: " + str((true_results/counter)*100) + "%")
            # for paraphrase = cosine > 0.2: 53.01282051282051%
            # for paraphrase = cosine > 0.3: 55.051282051282044%
            # for paraphrase = cosine > 0.4: 58.14102564102564%
            # for paraphrase = cosine > 0.5: 62.858974358974365%
            # for paraphrase = cosine > 0.6: 69.02564102564102%
            # for paraphrase = cosine > 0.7: 73.46153846153847%
            # for paraphrase = cosine > 0.8: 75.17948717948718%
            # for paraphrase = cosine > 0.9: 72.05128205128204%

            #print("True positives: " + str((true_positives/counter)*100) + "%")
            # for paraphrase = cosine > 0.2: 50.0%
            # for paraphrase = cosine > 0.3: 49.97435897435897%
            # for paraphrase = cosine > 0.4: 49.91025641025641%
            # for paraphrase = cosine > 0.5: 49.88461538461538%
            # for paraphrase = cosine > 0.6: 49.58974358974359%
            # for paraphrase = cosine > 0.7: 48.44871794871795%
            # for paraphrase = cosine > 0.8: 44.6025641025641%
            # for paraphrase = cosine > 0.9: 30.897435897435898%

            #print("True negatives: " + str((true_negatives/counter)*100) + "%")
            # for paraphrase = cosine > 0.2: 3.0128205128205128%
            # for paraphrase = cosine > 0.3: 5.076923076923077%
            # for paraphrase = cosine > 0.4: 8.23076923076923%
            # for paraphrase = cosine > 0.5: 12.974358974358974%
            # for paraphrase = cosine > 0.6: 19.435897435897438%
            # for paraphrase = cosine > 0.7: 25.01282051282051%
            # for paraphrase = cosine > 0.8: 30.57692307692308%
            # for paraphrase = cosine > 0.9: 41.15384615384615%

            #print("False results: " + str((false_results/counter)*100) + "%")
            # for paraphrase = cosine > 0.2: 46.98717948717949%
            # for paraphrase = cosine > 0.3: 44.94871794871795%
            # for paraphrase = cosine > 0.4: 41.85897435897436%
            # for paraphrase = cosine > 0.5: 37.14102564102564%
            # for paraphrase = cosine > 0.6: 30.974358974358974%
            # for paraphrase = cosine > 0.7: 26.53846153846154%
            # for paraphrase = cosine > 0.8: 24.82051282051282%
            # for paraphrase = cosine > 0.9: 27.94871794871795%

            #print("False positives: " + str((false_positives/counter)*100) + "%")
            # for paraphrase = cosine > 0.2: 46.98717948717949%
            # for paraphrase = cosine > 0.3: 44.92307692307692%
            # for paraphrase = cosine > 0.4: 41.76923076923077%
            # for paraphrase = cosine > 0.5: 37.02564102564103%
            # for paraphrase = cosine > 0.6: 30.564102564102562%
            # for paraphrase = cosine > 0.7: 24.987179487179485%
            # for paraphrase = cosine > 0.8: 19.423076923076923%
            # for paraphrase = cosine > 0.9: 8.846153846153847%

            #print("False negatives: " + str((false_negatives/counter)*100) + "%")
            # for paraphrase = cosine > 0.2: 0.0%
            # for paraphrase = cosine > 0.3: 0.02564102564102564%
            # for paraphrase = cosine > 0.4: 0.08974358974358974%
            # for paraphrase = cosine > 0.5: 0.11538461538461539%
            # for paraphrase = cosine > 0.6: 0.41025641025641024%
            # for paraphrase = cosine > 0.7: 1.5512820512820513%
            # for paraphrase = cosine > 0.8: 5.397435897435898%
            # for paraphrase = cosine > 0.9: 19.102564102564102%

            results_file.write("Precision: " + str(((true_positives)/(true_positives + false_positives))*100) + "%\n")
            #print("Precision: " + str(((true_positives) / (true_positives + false_positives))*100) + "%")
            # for paraphrase = cosine > 0.2: 51.55320555188367%
            # for paraphrase = cosine > 0.3: 52.66144285328289%
            # for paraphrase = cosine > 0.4: 54.439938470144035%
            # for paraphrase = cosine > 0.5: 57.39784629001328%
            # for paraphrase = cosine > 0.6: 61.8682021753039%
            # for paraphrase = cosine > 0.7: 65.97416201117319%
            # for paraphrase = cosine > 0.8: 69.66359631557869%
            # for paraphrase = cosine > 0.9: 77.74193548387098%

            results_file.write("Recall: " + str(((true_positives)/(true_positives + false_negatives))*100) + "%\n")
            #print("Recall: " + str(((true_positives)/(true_positives + false_negatives))*100) + "%")
            # for paraphrase = cosine > 0.2: 100.0%
            # for paraphrase = cosine > 0.3: 99.94871794871794%
            # for paraphrase = cosine > 0.4: 99.82051282051282%
            # for paraphrase = cosine > 0.5: 99.76923076923076%
            # for paraphrase = cosine > 0.6: 99.17948717948718%
            # for paraphrase = cosine > 0.7: 96.8974358974359%
            # for paraphrase = cosine > 0.8: 89.2051282051282%
            # for paraphrase = cosine > 0.9: 61.794871794871796%

            results_file.write("Specificity: " + str(((true_negatives) / (true_negatives + false_positives))*100) + "%\n")
            #print("Specificity: " + str(((true_negatives) / (true_negatives + false_positives))*100) + "%")
            # for paraphrase = cosine > 0.2: 6.0256410256410255%
            # for paraphrase = cosine > 0.3: 10.153846153846153%
            # for paraphrase = cosine > 0.4: 16.46153846153846%
            # for paraphrase = cosine > 0.5: 25.94871794871795%
            # for paraphrase = cosine > 0.6: 38.871794871794876%
            # for paraphrase = cosine > 0.7: 50.02564102564102%
            # for paraphrase = cosine > 0.8: 61.15384615384616%
            # for paraphrase = cosine > 0.9: 82.3076923076923%

            results_file.write("F-measure: " + str(f_measure) +"\n")
            results_file.write("\n")
            #print("F-measure: " + str(f_measure))
            # for paraphrase = cosine > 0.2: 0.680331443523768
            # for paraphrase = cosine > 0.3: 0.6897894178021589
            # for paraphrase = cosine > 0.4: 0.7045516242873948
            # for paraphrase = cosine > 0.5: 0.7287199175952804
            # for paraphrase = cosine > 0.6: 0.7620173364854215
            # for paraphrase = cosine > 0.7: 0.7850020772746157
            # for paraphrase = cosine > 0.8: 0.7823251630312569
            # for paraphrase = cosine > 0.9: 0.6885714285714286
