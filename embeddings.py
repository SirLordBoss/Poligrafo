import time
import numpy
import nltk

from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment


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
            file = open(source, "r")
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
    emb = Embeddings.loadEmbeddingsFromFile('D:\\Users\\ruica\\Projeto_FKNews')
    dt = time.time() - t0
    if emb is None:
        print("ERROR LOADING SOURCE FILE!")
    else:
        print('OK:.... %d    dt: %6.2f s' % (emb.length(), dt))

    sentence = [
        "the cat is in my office.",
        "the office has a cat.",
        "this was my first attempt to discover a new black hole.",
        "my bank will provide me with a new credit."
    ]

    for i in range(len(sentence)-1):
        print(sentence[i])
        vi = emb.getVectorSum(sentence[i])
        for j in range(i+1, len(sentence)):
            vj = emb.getVectorSum(sentence[j])
            sij = emb.cosim(vi, vj)
            print("%.5f <--- %s" % (sij, sentence[j]))
        print()

    print("%.5f" % emb.simBestConnect(sentence[0], sentence[1]))
