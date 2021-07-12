#xml parsing
#import requests
import xml.etree.ElementTree as ET

#sentiment detection
import spacy
from spacy import displacy
#from negspacy.negation import Negex
#from negspacy.termsets import termset

#ts = termset('en')
#prepping negspacy
nlp = spacy.load("en_core_web_sm")

tree = ET.parse('contradiction_dev_data.xml')

root = tree.getroot()

#print(root.tag)

#sentence_group = []

counter = 0

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

precision = 0
recall = 0
specificity = 0
f_measure = 0


def most_similar(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return by_similarity[:10]

def found_similar_verb(verb, verb_array):
    for verb_to_check in verb_array:
        if(nlp(verb_to_check).similarity(nlp(verb)) > 0.5):
            #print("found similar verb")
            return True
    return False

def get_similar_verb(verb, verb_array):
    for verb_to_check in verb_array:
        if(nlp(verb_to_check).similarity(nlp(verb)) > 0.5):
            return verb_to_check

def detected_verb_contradiction(sentence_1, sentence_2):
    sent1 = nlp(sentence_1)
    sent1_verbs = [tok for tok in sent1 if tok.pos_ == "VERB" or tok.pos_ == "AUX"]
    sent1_verb_lemmas = [tok.lemma_ for tok in sent1 if tok.pos_ == "VERB" or tok.pos_ == "AUX"]
    sent1_verb_children = [[child for child in token.children] for token in sent1_verbs]
    sent1_verb_and_children = dict(zip(sent1_verb_lemmas, sent1_verb_children))
    sent1_pobj = [tok for tok in sent1 if tok.dep_ == "pobj"]

    print("sentence 1 verbs and children: " + str(sent1_verb_and_children))

    sent2 = nlp(sentence_2)
    sent2_verbs = [tok for tok in sent2 if tok.pos_ == "VERB" or tok.pos_ == "AUX"]
    sent2_verb_lemmas = [tok.lemma_ for tok in sent2 if tok.pos_ == "VERB" or tok.pos_ == "AUX"]
    sent2_verb_children = [[child for child in token.children] for token in sent2_verbs]
    sent2_verb_and_children = dict(zip(sent2_verb_lemmas, sent2_verb_children))
    sent2_pobj = [tok for tok in sent2 if tok.dep_ == "pobj"]

    print("sentence 2 verbs and children: " + str(sent2_verb_and_children))

    for verb, children in sent1_verb_and_children.items():
        print("verb: " + str(verb) + ", child: " + str(children))
        if verb in sent2_verb_and_children.keys():
            print("found common verb!")
            for child in sent2_verb_and_children[verb]:
                print("child: " + str(child) + "(dep: " + str(child.dep_) + ")")
                if(child.dep_ == "neg"):
                    return True
        elif found_similar_verb(verb, sent2_verb_and_children.keys()):
            print("found similar verb! " + verb + " -> " + get_similar_verb(verb, sent2_verb_and_children.keys()))
            for child in sent2_verb_and_children[get_similar_verb(verb, sent2_verb_and_children.keys())]:
                print("child: " + str(child) + "(dep: " + str(child.dep_) + ")")
                if(child.dep_ == "neg"):
                    return True
             # COMPUTE VERB SIMILARITY HERE
    #para cada verbo v na sentence 1, obter o objeto (dobj)
    #   para cada ocorrência de v em sentence 2
    #       ler objeto de v em sentence 2,
    #           se v estiver negado e os objetos dos verbos nas duas frases forem os mesmos, é contradição, e devolve true
    #           
    # se chegar ao final sem detetar contradição. devolve False

    return False



for child in root:
    print("\nchild attribute: " + str(child.attrib))
    counter += 1

    sentence_group = []
    for sentence in child:
        sentence_group.append(sentence.text)
        #print(sentence.text)

    print(sentence_group)

    sent1 = nlp(sentence_group[0])

    print("Sentence 1: " + sentence_group[0])
    sent1_tokens = [tok for tok in sent1]
    sent1_verbs = [tok.lemma_ for tok in sent1 if tok.pos_ == "VERB" or tok.pos_ == "AUX"]
    sent1_chunks = [chunk for chunk in sent1.noun_chunks]
    sent1_negation_tokens = [tok for tok in sent1 if tok.dep_ == 'neg']
    sent1_negation_token_heads = [token.head.lemma_ for token in sent1_negation_tokens]
    #print("sentence 1 tokens: " + str(sent1_tokens))
    #print("sentence 1 verbs: " + str(sent1_verbs))
    #print("sentence 1 chunks: " + str(sent1_chunks))
    #print("negation tokens: " + str(sent1_negation_tokens) + " -> heads: " + str(sent1_negation_token_heads))
    #options = {'compact': True, 'color': 'black', 'font': 'Arial'}
    #displacy.serve(sent1, style='dep', options=options)

    sent2 = nlp(sentence_group[1])

    print("Sentence 2: " + sentence_group[1])
    sent2_tokens = [tok for tok in sent2]
    sent2_chunks = [chunk for chunk in sent2.noun_chunks]
    sent2_negation_tokens = [tok for tok in sent2 if tok.dep_ == 'neg']
    sent2_negation_token_heads = [token.head for token in sent2_negation_tokens]
    sent2_negation_token_head_lemmas = [token.head.lemma_ for token in sent2_negation_tokens]
    #print("sentence 2 tokens: " + str(sent2_tokens))
    #print("sentence 2 chunks: " + str(sent2_chunks))
    #print("negation tokens: " + str(sent2_negation_tokens) + " -> heads: " + str(sent2_negation_token_head_lemmas) + "\n")

    if(detected_verb_contradiction(sentence_group[0], sentence_group[1]) or detected_verb_contradiction(sentence_group[1], sentence_group[0])):
        print("contradiction found")
        if(child.attrib['contradiction'] == 'YES'):
            print("And it's true!")
            true_positives += 1
        else:
            print("But it's fake")
            false_positives += 1
    else:
        print("no contradiction found")
        if(child.attrib['contradiction'] == 'YES'):
            print("but it was")
            false_negatives += 1
        else:
            print("and it wasn't!")
            true_negatives += 1


    #for neg_token_head in sent2_negation_token_heads:
    #    if(neg_token_head.pos_ == "VERB" or neg_token_head.pos_ == "AUX") and neg_token_head.lemma_ in sent1_verbs: #and neg_token_head.children == sent1_verbs.children:
    #        print("we have a contradiction")
    #        if(child.attrib['contradiction'] == 'YES'):
    #            print("And it's true!")
    #            true_positives += 1
    #        else:
    #            print("But it's fake")
    #            false_positives += 1
    #    else:
    #        print("no contradiction found")
    #        if(child.attrib['contradiction'] == 'YES'):
    #            print("but it was")
    #            false_negatives += 1
    #        else:
    #            print("and it wasn't!")
    #            true_negatives += 1

print("\nsentences: " + str(counter))

print("True positives: " + str(true_positives))
print("true negatives: " + str(true_negatives))
print("false positives: " + str(false_positives))
print("false negatives: " + str(false_negatives))

precision = true_positives/(true_positives + false_positives)

recall = true_positives/(true_positives + false_negatives)

specificity = true_negatives/(true_negatives + false_positives)

f_measure = 2 * ((precision * recall)/(precision + recall))

print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("Specificity: " + str(specificity))
print("f-measure: " + str(f_measure))

#########base result
#True positives: 32
#true negatives: 12
#false positives: 34
#false negatives: 17
#Precision: 0.48484848484848486
#Recall: 0.6530612244897959
#Specificity: 0.2608695652173913
#f-measure: 0.5565217391304348

####### with checking the verbs and lemmas
#true positives: 11
#true negatives: 34
#false positives: 12
#false negatives: 38
#Precision: 0.4782608695652174
#Recall: 0.22448979591836735
#Specificity: 0.7391304347826086
#f-measure: 0.3055555555555556

######  checking verbs, lemmas and auxiliaries



###### with the functions checking the sentences symetrically
#True positives: 206
#true negatives: 286
#false positives: 162
#false negatives: 268
#Precision: 0.5597826086956522
#Recall: 0.4345991561181435
#Specificity: 0.6383928571428571
#F-measure: 0.489311163895487

##### checking for verb similarity
#sentences: 922
#True positives: 201
#true negatives: 249
#false positives: 160
#false negatives: 216
#Precision: 0.556786703601108
#Recall: 0.48201438848920863
#Specificity: 0.60880195599022
#f-measure: 0.5167095115681235

###similarity > 0.8
#sentences: 922
#True positives: 215
#true negatives: 275
#false positives: 173
#false negatives: 259
#Precision: 0.5541237113402062
#Recall: 0.45358649789029537
#Specificity: 0.6138392857142857
#f-measure: 0.4988399071925754


###similarity > 0.7
#sentences: 922
#True positives: 269
#true negatives: 213
#false positives: 235
#false negatives: 205
#Precision: 0.5337301587301587
#Recall: 0.5675105485232067
#Specificity: 0.47544642857142855
#f-measure: 0.5501022494887525

###similarity > 0.6
#sentences: 922
#True positives: 314
#true negatives: 159
#false positives: 289
#false negatives: 160
#Precision: 0.5207296849087893
#Recall: 0.6624472573839663
#Specificity: 0.3549107142857143
#f-measure: 0.5831012070566388

###similarity > 0.5
#True positives: 333
#true negatives: 144
#false positives: 304
#false negatives: 141
#Precision: 0.5227629513343799
#Recall: 0.7025316455696202
#Specificity: 0.32142857142857145
#f-measure: 0.5994599459945994

###similarity > 0.4
#sentences: 922
#True positives: 312
#true negatives: 154
#false positives: 294
#false negatives: 162
#Precision: 0.5148514851485149
#Recall: 0.6582278481012658
#Specificity: 0.34375
#f-measure: 0.5777777777777777

#########TO DO:
#create better dataset
#detetar sinonimos de verbos
#relatorio
