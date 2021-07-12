from paraphrases import msrpcx

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

#heavily inspired by https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"
])

print(embeddings)

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

    similarity = cosine(embed([string_a])[0], embed([string_b])[0])

    print("similarity: ", similarity)

    if(similarity > 0.8):
        print("detected paraphrase")
        if(paraphrase == 1):
            print("true result")
            true_results += 1
            true_positives += 1
        else:
            print("false result")
            false_results += 1
            false_positives += 1
    else:
        print("did not detect paraphrase")
        if(paraphrase == 1):
            print("false result")
            false_results += 1
            false_negatives += 1
        else:
            print("true result")
            true_results += 1
            true_negatives += 1

    print()
    counter += 1

precision = true_positives/(true_positives + false_positives)

recall = true_positives/(true_positives + false_negatives)

specificity = true_negatives/(true_negatives + false_positives)

f_measure = 2 * ((precision * recall)/(precision + recall))

print("Final analysis")

print("True results: " + str((true_results/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 72.03846153846155%
# for paraphrase = cosine > 0.3: 72.97435897435898%
# for paraphrase = cosine > 0.4: 73.47435897435898%
# for paraphrase = cosine > 0.5: 74.44871794871794%
# for paraphrase = cosine > 0.6: 75.8076923076923%
# for paraphrase = cosine > 0.7: 76.02564102564102%
# for paraphrase = cosine > 0.8: 69.47435897435898%

print("True positives: " + str((true_positives/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 50.0%
# for paraphrase = cosine > 0.3: 49.96153846153846%
# for paraphrase = cosine > 0.4: 49.82051282051282%
# for paraphrase = cosine > 0.5: 49.11538461538461%
# for paraphrase = cosine > 0.6: 46.14102564102564%
# for paraphrase = cosine > 0.7: 39.243589743589745%
# for paraphrase = cosine > 0.8: 25.743589743589745%

print("True negatives: " + str((true_negatives/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 22.038461538461537%
# for paraphrase = cosine > 0.3: 23.01282051282051%
# for paraphrase = cosine > 0.4: 23.653846153846153%
# for paraphrase = cosine > 0.5: 25.333333333333336%
# for paraphrase = cosine > 0.6: 29.666666666666668%
# for paraphrase = cosine > 0.7: 36.782051282051285%
# for paraphrase = cosine > 0.8: 43.730769230769226%

print("False results: " + str((false_results/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 27.96153846153846%
# for paraphrase = cosine > 0.3: 27.025641025641022%
# for paraphrase = cosine > 0.4: 26.52564102564103%
# for paraphrase = cosine > 0.5: 25.551282051282055%
# for paraphrase = cosine > 0.6: 24.192307692307693%
# for paraphrase = cosine > 0.7: 23.974358974358974%
# for paraphrase = cosine > 0.8: 30.525641025641026%

print("False positives: " + str((false_positives/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 27.96153846153846%
# for paraphrase = cosine > 0.3: 26.987179487179485%
# for paraphrase = cosine > 0.4: 26.346153846153847%
# for paraphrase = cosine > 0.5: 24.666666666666668%
# for paraphrase = cosine > 0.6: 20.333333333333332%
# for paraphrase = cosine > 0.7: 13.217948717948719%
# for paraphrase = cosine > 0.8: 6.269230769230768%

print("False negatives: " + str((false_negatives/counter)*100) + "%")
# for paraphrase = cosine > 0.2: 0.0%
# for paraphrase = cosine > 0.3: 0.038461538461538464%
# for paraphrase = cosine > 0.4: 0.1794871794871795%
# for paraphrase = cosine > 0.5: 0.8846153846153846%
# for paraphrase = cosine > 0.6: 3.858974358974359%
# for paraphrase = cosine > 0.7: 10.756410256410257%
# for paraphrase = cosine > 0.8: 24.25641025641026%

print("Precision: " + str(((true_positives) / (true_positives + false_positives))*100) + "%")
# for paraphrase = cosine > 0.2: 64.13418845584607%
# for paraphrase = cosine > 0.3: 64.92835721426191%
# for paraphrase = cosine > 0.4: 65.40986365931661%
# for paraphrase = cosine > 0.5: 66.56820156385751%
# for paraphrase = cosine > 0.6: 69.41176470588235%
# for paraphrase = cosine > 0.7: 74.80449657869013%
# for paraphrase = cosine > 0.8: 80.4164997997597%

print("Recall: " + str(((true_positives)/(true_positives + false_negatives))*100) + "%")
# for paraphrase = cosine > 0.2: 100.0%
# for paraphrase = cosine > 0.3: 99.92307692307692%
# for paraphrase = cosine > 0.4: 99.64102564102564%
# for paraphrase = cosine > 0.5: 98.23076923076923%
# for paraphrase = cosine > 0.6: 92.28205128205128%
# for paraphrase = cosine > 0.7: 78.48717948717949%
# for paraphrase = cosine > 0.8: 51.48717948717949%

print("Specificity: " + str(((true_negatives) / (true_negatives + false_positives))*100) + "%")
# for paraphrase = cosine > 0.2: 44.07692307692307%
# for paraphrase = cosine > 0.3: 46.02564102564102%
# for paraphrase = cosine > 0.4: 47.30769230769231%
# for paraphrase = cosine > 0.5: 50.66666666666667%
# for paraphrase = cosine > 0.6: 59.333333333333336%
# for paraphrase = cosine > 0.7: 73.56410256410257%
# for paraphrase = cosine > 0.8: 87.46153846153845%

print("F-measure: " + str(f_measure))
# for paraphrase = cosine > 0.2: 0.7814848211602043
# for paraphrase = cosine > 0.3: 0.7871137144011312
# for paraphrase = cosine > 0.4: 0.7897571385021848
# for paraphrase = cosine > 0.5: 0.7935784567581564
# for paraphrase = cosine > 0.6: 0.7922949917446341
# for paraphrase = cosine > 0.7: 0.766016016016016
# for paraphrase = cosine > 0.8: 0.6277942785680788
