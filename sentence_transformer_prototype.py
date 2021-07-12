from paraphrases import msrpcx

from sentence_transformers import SentenceTransformer, util
import numpy as np

#https://www.sbert.net/examples/applications/semantic-search/README.html

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

counter = 0

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

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

for paraphrase, string_a, string_b in msrpcx:
    print(counter)
    print(paraphrase)
    print(string_a)
    #print(embed([string_a])[0])
    print(string_b)
    #print(embed([string_b])[0])

    similarity = cosine(model.encode(string_a), model.encode(string_b))

    #print("similarity: ", similarity)

    if(similarity > 0.2):
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

print("Final analysis")

print("True results: " + str((true_results/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 71.41025641025641%
# for paraphrase = cosine > 0.3: 72.97435897435898%
# for paraphrase = cosine > 0.4: 74.1025641025641%
# for paraphrase = cosine > 0.5: 76.26923076923077%
# for paraphrase = cosine > 0.6: 78.08974358974359%
# for paraphrase = cosine > 0.7: 77.94871794871796%
# for paraphrase = cosine > 0.8: 72.07692307692307%

print("True positives: " + str((true_positives/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 50.0%
# for paraphrase = cosine > 0.3: 50.0%
# for paraphrase = cosine > 0.4: 49.91025641025641%
# for paraphrase = cosine > 0.5: 49.358974358974365%
# for paraphrase = cosine > 0.6: 47.23076923076923%
# for paraphrase = cosine > 0.7: 41.52564102564102%
# for paraphrase = cosine > 0.8: 30.141025641025642%

print("True negatives: " + str((true_negatives/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 21.41025641025641%
# for paraphrase = cosine > 0.3: 22.974358974358974%
# for paraphrase = cosine > 0.4: 24.192307692307693%
# for paraphrase = cosine > 0.5: 26.910256410256412%
# for paraphrase = cosine > 0.6: 30.85897435897436%
# for paraphrase = cosine > 0.7: 36.42307692307693%
# for paraphrase = cosine > 0.8: 41.93589743589744%

print("False results: " + str((false_results/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 28.589743589743588%
# for paraphrase = cosine > 0.3: 27.025641025641022%
# for paraphrase = cosine > 0.4: 25.8974358974359%
# for paraphrase = cosine > 0.5: 23.73076923076923%
# for paraphrase = cosine > 0.6: 21.91025641025641%
# for paraphrase = cosine > 0.7: 22.05128205128205%
# for paraphrase = cosine > 0.8: 27.923076923076923%

print("False positives: " + str((false_positives/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 28.589743589743588%
# for paraphrase = cosine > 0.3: 27.025641025641022%
# for paraphrase = cosine > 0.4: 25.80769230769231%
# for paraphrase = cosine > 0.5: 23.089743589743588%
# for paraphrase = cosine > 0.6: 19.141025641025642%
# for paraphrase = cosine > 0.7: 13.576923076923078%
# for paraphrase = cosine > 0.8: 8.064102564102564%

print("False negatives: " + str((false_negatives/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 0.0%
# for paraphrase = cosine > 0.3: 0.0%
# for paraphrase = cosine > 0.4: 0.08974358974358974%
# for paraphrase = cosine > 0.5: 0.641025641025641%
# for paraphrase = cosine > 0.6: 2.769230769230769%
# for paraphrase = cosine > 0.7: 8.474358974358974%
# for paraphrase = cosine > 0.8: 19.85897435897436%

print("Precision: " + str(((true_positives)/(true_positives + false_positives))*100) + "%")
# for paraphrase = cosine > 0.2: 63.621533442088094%
# for paraphrase = cosine > 0.3: 64.91344873501997%
# for paraphrase = cosine > 0.4: 65.91601760921097%
# for paraphrase = cosine > 0.5: 68.12953459564679%
# for paraphrase = cosine > 0.6: 71.1609039984547%
# for paraphrase = cosine > 0.7: 75.36063285248953%
# for paraphrase = cosine > 0.8: 78.89261744966443%

print("Recall: " + str(((true_positives)/(true_positives + false_negatives))*100) + "%")
# for paraphrase = cosine > 0.2: 100.0%
# for paraphrase = cosine > 0.3: 100.0%
# for paraphrase = cosine > 0.4: 99.82051282051282%
# for paraphrase = cosine > 0.5: 98.71794871794873%
# for paraphrase = cosine > 0.6: 94.46153846153847%
# for paraphrase = cosine > 0.7: 83.05128205128204%
# for paraphrase = cosine > 0.8: 60.282051282051285%

print("Specificity: " + str(((true_negatives)/(true_negatives + false_positives))*100) + "%")
# for paraphrase = cosine > 0.2: 42.82051282051282%
# for paraphrase = cosine > 0.3: 45.94871794871795%
# for paraphrase = cosine > 0.4: 48.38461538461539%
# for paraphrase = cosine > 0.5: 53.820512820512825%
# for paraphrase = cosine > 0.6: 61.71794871794872%
# for paraphrase = cosine > 0.7: 72.84615384615385%
# for paraphrase = cosine > 0.8: 83.87179487179488%

print("F-measure: " + str(f_measure))
# for paraphrase = cosine > 0.2: 0.7776669990029911
# for paraphrase = cosine > 0.3: 0.7872426322163908
# for paraphrase = cosine > 0.4: 0.7940036712217009
# for paraphrase = cosine > 0.5: 0.8061983038425297
# for paraphrase = cosine > 0.6: 0.8117219345598766
# for paraphrase = cosine > 0.7: 0.7901927299341303
# for paraphrase = cosine > 0.8: 0.6834302325581395