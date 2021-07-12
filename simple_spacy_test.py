import spacy 

nlp = spacy.load("en_core_web_sm")

sent1 = nlp(
    "The American Reform Party is a minor political party in the United States that was formed in a factional split from the larger Reform Party of the United States in October 1997.")
sent1_verbs = [tok for tok in sent1 if tok.pos_ == "VERB" or tok.pos_ == "AUX"]
sent1_verb_children = [[child for child in token.children] for token in sent1_verbs]
sent1_verb_and_children = list(zip(sent1_verbs, sent1_verb_children))
sent1_pobj = [token for token in sent1 if token.dep_ == "pobj"]
sent1_chunks = [chunk for chunk in sent1.noun_chunks]


sent2 = nlp(
    "The American Reform Party is not a political party, they do not have ballot access in any state, and they do not run candidates.")
sent2_verbs = [tok for tok in sent2 if tok.pos_ == "VERB" or tok.pos_ == "AUX"]
sent2_verb_children = [[child for child in token.children] for token in sent2_verbs]
sent2_pobj = [token for token in sent2 if token.dep_ == "pobj"]
sent2_chunks = [chunk for chunk in sent2.noun_chunks]


print('DEPENDENCY RELATIONS')
print('Key: ')
print('TEXT, DEP, HEAD_TEXT, HEAD_POS, CHILDREN')

print("Sentence 1")
for token in sent1:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])
print("obj: " + str(sent1_pobj))
print("noun chunks: " + str(sent1_chunks))
print("verbs: " + str(sent1_verbs))
print("verb children: " + str(sent1_verb_children))
print("verbs and children: " + str(sent1_verb_and_children))

print("\nsentence 2")
for token in sent2:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])
print("obj: " + str(sent2_pobj))
print("noun chunks: " + str(sent2_chunks))
print("verbs: " + str(sent2_verbs))
print("verb children: " + str(sent2_verb_children))
