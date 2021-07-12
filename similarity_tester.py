import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})

def most_similar(word):
    queries = [w for w in word.vocab if w.prob >= -15]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return by_similarity[:10]

#print([w.lower_ for w in most_similar(nlp.vocab[u'dog'])])

token = nlp("give")[0]

print(nlp('give').similarity(nlp('gift')))
