from paraphrases import msrpcx

from sentence_transformers import SentenceTransformer, util
import numpy as np

import matplotlib.pyplot as plt

#https://www.sbert.net/examples/applications/semantic-search/README.html

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

counter = 0

similarity_vec = []

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

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

for paraphrase, string_a, string_b in msrpcx:
    #print(paraphrase)
    #print(string_a)
    #print(embed([string_a])[0])
    #print(string_b)
    #print(embed([string_b])[0])

    similarity_vec.append([cosine(model.encode(string_a), model.encode(string_b)), paraphrase])
    counter += 1
    print("similarity: " + str(counter) + ", paraphrase: " + str(paraphrase))

x_axis = []
y_axis = []

with open("results_sentence_transformer.txt", "w") as results_file:
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

        results_file.write("Precision: " + str(((true_positives) / (true_positives + false_positives))*100) + "%\n")
        #print("Precision: " + str(((true_positives) / (true_positives + false_positives))*100) + "%")

        results_file.write("Recall: " + str(((true_positives) / (true_positives + false_negatives))*100) + "%\n")
        #print("Recall: " + str(((true_positives) / (true_positives + false_negatives))*100) + "%")

        results_file.write("Specificity: " + str(((true_negatives) / (true_negatives + false_positives))*100) + "%\n")
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
    plt.title('SentenceTransformer')

    # function to show the plot, after saving it
    plt.savefig('sentence_transformer_graph.png')
    plt.show()
    

