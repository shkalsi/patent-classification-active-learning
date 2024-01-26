import numpy as np
import pandas as pd
import json
import csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict

#Based on the paper
ALPHA = 100
BETA = 350
DELTA = 5
GAMMA = 286

def sample(S, n):
    #print(S)
    random.shuffle(S)
    #print(S)
    return S[:int(n)]

def getBasedOnEntropy(entropy, n):
    entropy = OrderedDict(sorted(entropy.items(), key=lambda item: item[1], reverse=True))
    desc_sorted_pns = list(entropy.keys())
    return desc_sorted_pns[:int(n)]

def main():
    filepath = "/home/skalsi@minesoft.local/data/patent_classification/data/quantum-qubit-generation_merged_family_id_float.tsv"
    precs = []
    recs = []
    fs = []

    runs = 200
    res_p = np.zeros((runs, 11))
    res_r = np.zeros((runs, 11))
    res_f1 = np.zeros((runs, 11))

    for i in range(runs):
        gold_standard = pd.read_csv(filepath, sep= '\t', converters={'doc_vector': pd.eval})
        family_ids = gold_standard['DocDB Family ID'].unique()

        gold_standard_positive = (gold_standard.loc[gold_standard['Class'] == "positive"]).drop(["Unnamed: 0"], axis=1)
        gold_standard_negative = (gold_standard.loc[gold_standard['Class'] == "negative"]).drop(["Unnamed: 0"], axis=1)

        hold_out = sample(gold_standard_positive.values.tolist(), (len(gold_standard_positive.index)/len(gold_standard.index))*GAMMA)
        hold_out = hold_out + (sample(gold_standard_negative.values.tolist(), (len(gold_standard_negative.index)/len(gold_standard.index))*GAMMA))
        hold_out_df = pd.DataFrame(hold_out, columns=["Serial no.", "Class", "doc_vector", "DocDB Family ID"])
        
        #T = G\H
        training_df_init_posi = gold_standard_positive[ ~gold_standard_positive["Serial no."].isin(hold_out_df["Serial no."]) ]#.drop(["Unnamed: 0"], axis=1)
        training_df_init_neg = gold_standard_negative[ ~gold_standard_negative["Serial no."].isin(hold_out_df["Serial no."]) ]#.drop(["Unnamed: 0"], axis=1)
        training_set = sample(training_df_init_posi.values.tolist(), ALPHA/2) + sample(training_df_init_neg.values.tolist(), ALPHA/2)
        training_df = pd.DataFrame(training_set, columns=["Serial no.", "Class", "doc_vector", "DocDB Family ID"])

        vectors = {}
        for index, row in gold_standard.iterrows():
            vectors[row["Serial no."]] = row["doc_vector"]
        
        precisions = []
        recalls = []
        f1s = []
        training_set_size = len(training_df.index)
        j = 0
        rf = RandomForestClassifier(criterion="entropy")
        y_coord = 0
        while training_set_size <= BETA:
            print(j)
            train = training_df['DocDB Family ID'].unique()

            #E = G\T
            eval_df = gold_standard[ ~gold_standard["Serial no."].isin(training_df["Serial no."]) ].drop(["Unnamed: 0"], axis=1)
            evals = eval_df['DocDB Family ID'].unique()

            train_pns = training_df[training_df['DocDB Family ID'].isin(train)]['Serial no.'].tolist()
            eval_pns = eval_df[eval_df['DocDB Family ID'].isin(evals)]['Serial no.'].tolist()
            X_train = np.array([vectors[pn] for pn in train_pns]).squeeze()
            X_test = np.array([vectors[pn] for pn in eval_pns]).squeeze()
            y_train = [0 if any(training_df[training_df['Serial no.'] == pn]['Class'] == 'negative') else 1 for pn in train_pns]
            y_test = [0 if any(eval_df[eval_df['Serial no.'] == pn]['Class'] == 'negative') else 1 for pn in eval_pns]
            #print(len(y_train))

            rf.fit(X_train, y_train)

            predictions = rf.predict(X_test)
            p = precision_score(y_test, predictions)
            precisions.append(p)

            r = recall_score(y_test, predictions)
            recalls.append(r)

            f1 = f1_score(y_test, predictions)
            f1s.append(f1)

            eval_df_positive = eval_df.loc[eval_df['Class'] == "positive"]
            eval_df_negative = eval_df.loc[eval_df['Class'] == "negative"]
            if p >= r:
                delta_positive = eval_df_positive[ ~eval_df_positive["Serial no."].isin(hold_out_df["Serial no."]) ]
                delta_positive_pns = delta_positive[delta_positive['DocDB Family ID'].isin(evals)]['Serial no.'].tolist()
                entropy = {}
                for pn in delta_positive_pns:
                    X_delta_positive = np.array(vectors[pn]).squeeze()
                    y_delta_positive = 1
                    if any(delta_positive[delta_positive['Serial no.'] == pn]['Class'] == 'negative'):
                        y_delta_positive = 0
                    proba_predictions = rf.predict_proba(X_delta_positive.reshape(1, -1))
                    entropy[pn] = -np.sum(proba_predictions * np.log2(proba_predictions, where=proba_predictions>0))   

                delta_positive_top_pns = getBasedOnEntropy(entropy, DELTA)
                delta_positive_top_delta = delta_positive[delta_positive["Serial no."].isin(delta_positive_top_pns)].values.tolist()
                training_set = training_set + (delta_positive_top_delta)
            else:
                delta_negative = eval_df_negative[ ~eval_df_negative["Serial no."].isin(hold_out_df["Serial no."]) ]
                delta_negative_pns = delta_negative[delta_negative['DocDB Family ID'].isin(evals)]['Serial no.'].tolist()
                entropy = {}
                for pn in delta_negative_pns:
                    X_delta_negative = np.array(vectors[pn]).squeeze()
                    y_delta_negative = 1
                    if any(delta_negative[delta_negative['Serial no.'] == pn]['Class'] == 'negative'):
                        y_delta_negative = 0
                    proba_predictions = rf.predict_proba(X_delta_negative.reshape(1, -1))
                    entropy[pn] = -np.sum(proba_predictions * np.log2(proba_predictions))

                delta_negative_top_pns = getBasedOnEntropy(entropy, DELTA)
                delta_negative_top_delta = delta_negative[delta_negative["Serial no."].isin(delta_negative_top_pns)].values.tolist()
                training_set = training_set + (delta_negative_top_delta)
            
            training_df = pd.DataFrame(training_set, columns=["Serial no.", "Class", "doc_vector", "DocDB Family ID"])
            training_set_size = len(training_df.index)

            if j%5==0:
                print("results:")
                test = hold_out_df['DocDB Family ID'].unique()
                test_pns = hold_out_df[hold_out_df['DocDB Family ID'].isin(test)]['Serial no.'].tolist()
                X_test = np.array([vectors[pn] for pn in test_pns]).squeeze()
                y_test = [0 if any(hold_out_df[hold_out_df['Serial no.'] == pn]['Class'] == 'negative') else 1 for pn in test_pns]

                predictions = rf.predict(X_test)
                test_p = precision_score(y_test, predictions)
                test_r = recall_score(y_test, predictions)
                test_f1 = f1_score(y_test, predictions)
                print(test_p)
                print(test_r)
                print(test_f1)
                print("\n\n")

                res_p[i][y_coord] = test_p
                res_r[i][y_coord] = test_r
                res_f1[i][y_coord] = test_f1

                y_coord += 1
            
            j+=1
        

    precision = np.mean(res_p, axis=0)
    recall = np.mean(res_r, axis=0)
    f1 = np.mean(res_f1, axis=0)

    print(precision)
    print(recall)
    print(f1)

    print(precision[-1])
    print(recall[-1])
    print(f1[-1])

main()