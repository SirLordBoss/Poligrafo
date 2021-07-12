#xml parsing
#import requests
import xml.etree.ElementTree as ET

#sentiment detection
import spacy
from spacy import displacy
from negspacy.negation import Negex
from negspacy.termsets import termset

ts = termset('en')
#prepping negspacy
nlp = spacy.load("en_core_sci_lg")
#negex = Negex(nlp, ent_types=["PERSON", "ORG"])

nlp.add_pipe("negex", last=True)

#nlp.add_pipe("negex", config={"ent_types": ["PERSON", "ORG"]})
#artigo sobre entity types
#https://towardsdatascience.com/named-entity-recognition-ner-using-spacy-nlp-part-4-28da2ece57c6 

#nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})
#

tree = ET.parse('contradiction_dev_data.xml')

root = tree.getroot()

print(root.tag)

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

for child in root:
    print("child attribute: " + str(child.attrib))
    counter += 1
    
    sentence_group = []
    for sentence in child:
        sentence_group.append(sentence.text)
        #print(sentence.text)

    print(sentence_group)

    sent1 = nlp(sentence_group[0])
    sent1_entities = []
    sent1_negexes = []
    sent1_data = []

    print("\nSentence 1:")
    for s1_e in sent1.ents:
        #print("Entity text: " + str(s1_e.text) + "\nEntity negated: " + str(s1_e._.negex))
        sent1_entities.append(s1_e.text)
        sent1_negexes.append(s1_e._.negex)
        #sent1_data.append([s1_e.text, s1_e.negex])

    sent1_data = dict(zip(sent1_entities, sent1_negexes))
    print("sentence 1 entities: " + str(sent1_entities))
    print("sentence 1 dictionary: " + str(sent1_data))

    sent2 = nlp(sentence_group[1])
    sent2_entities = []
    sent2_negexes = []
    sent2_data = []

    print("\nSentence 2:")
    for s2_e in sent2.ents:
        #print("Entity text: " + str(s2_e.text) + "\nEntity negated: " + str(s2_e._.negex))
        sent2_entities.append(s2_e.text)
        sent2_negexes.append(s2_e._.negex)
        #sent1_data.append([s1_e.text, s1_e.negex])

    sent2_data = dict(zip(sent2_entities, sent2_negexes))
    print("sentence 2 entities: " + str(sent2_entities))
    print("sentence 2 dictionary: " + str(sent2_data))

    matches = 0
    failed_matches = 0

    for entity in sent1_entities:
        #print("entity: " + str(entity))
        if(entity in sent2_entities):
            if(sent1_data[entity] == sent2_data[entity]):
                print("sentence 1 entity (" + str(entity) + "): " + str(sent1_data[entity]) +
                         " -> sentence 2 entity (" + str(entity) + "): " + str(sent2_data[entity]))
                matches += 1
                print("that's a match")
            else:
                print("sentence 1 entity (" + str(entity) + "): " + str(sent1_data[entity]) +
                      " -> sentence 2 entity (" + str(entity) + "): " + str(sent2_data[entity]))
                failed_matches += 1
                print("not a match")
    
    print("matches: " + str(matches) + ", failed matches: " + str(failed_matches))
    if(failed_matches >= 1):
        print("we have a contradiction")
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
            false_negatives +=1
        else:
            print("and it wasn't!")
            true_negatives += 1

print("sentences: " + str(counter))

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

########### without adding ent_types
### if matches > failed_matches
#True positives: 7
#true negatives: 47
#false positives: 4
#false negatives: 44
#Precision: 0.6363636363636364
#Recall: 0.13725490196078433
#Specificity: 0.9215686274509803
#f-measure: 0.22580645161290325

### if failed_matches > matches
#True positives: 13
#true negatives: 37
#false positives: 14
#false negatives: 38
#Precision: 0.48148148148148145
#Recall: 0.2549019607843137
#Specificity: 0.7254901960784313
#f-measure: 0.3333333333333333

###if failed_matches > 1
#True positives: 223
#true negatives: 263
#false positives: 185
#false negatives: 251
#Precision: 0.5465686274509803
#Recall: 0.4704641350210971
#Specificity: 0.5870535714285714
#f-measure: 0.5056689342403627

########### with nlp.add_pipe("negex", config={"ent_types": ["PERSON", "ORG"]})
###if matches > failed_matches
#True positives: 22
#true negatives: 36
#false positives: 15
#false negatives: 29
#Precision: 0.5945945945945946
#Recall: 0.43137254901960786
#Specificity: 0.7058823529411765
#f-measure: 0.5000000000000001

###if failed_matches > matches
#True positives: 5
#true negatives: 46
#false positives: 5
#false negatives: 46
#Precision: 0.5
#Recall: 0.09803921568627451
#Specificity: 0.9019607843137255
#f-measure: 0.1639344262295082

###if failed_matches > 1
#True positives: 11
#true negatives: 42
#false positives: 9
#false negatives: 41
#Precision: 0.55
#Recall: 0.21153846153846154
#Specificity: 0.8235294117647058
#f-measure: 0.3055555555555555

########### with nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()}) - NO EFFECT
### if matches > failed_matches
#True positives: 7
#true negatives: 47
#false positives: 4
#false negatives: 44
#Precision: 0.6363636363636364
#Recall: 0.13725490196078433
#Specificity: 0.9215686274509803
#f-measure: 0.22580645161290325

###if failed_matches > matches
#True positives: 13
#true negatives: 37
#false positives: 14
#false negatives: 38
#Precision: 0.48148148148148145
#Recall: 0.2549019607843137
#Specificity: 0.7254901960784313
#f-measure: 0.3333333333333333

########### with en_core_sci_lg - WORSE
### if matches > failed_matches
#True positives: 10
#true negatives: 40
#false positives: 11
#false negatives: 41
#Precision: 0.47619047619047616
#Recall: 0.19607843137254902
#Specificity: 0.7843137254901961
#f-measure: 0.2777777777777778

###if failed_matches > matches
#True positives: 15
#true negatives: 31
#false positives: 20
#false negatives: 36
#Precision: 0.42857142857142855
#Recall: 0.29411764705882354
#Specificity: 0.6078431372549019
#f-measure: 0.3488372093023256

###if failed_matches > 1
#True positives: 27
#true negatives: 24
#false positives: 27
#false negatives: 25
#Precision: 0.5
#Recall: 0.5192307692307693
#Specificity: 0.47058823529411764
#f-measure: 0.509433962264151


#########TO DO:
#Negação de verbos
#relatorio
