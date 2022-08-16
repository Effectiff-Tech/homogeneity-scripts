import argparse
import os
import time
from docx2python import docx2python
from blingfire import *
import readability
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.preprocessing import StandardScaler as SS # z-score standardization
from sklearn.cluster import KMeans, DBSCAN # clustering algorithms
from sklearn.decomposition import PCA # dimensionality reduction
from kneed import KneeLocator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import shutil
from itertools import combinations
import glob
from extraction_functions import individual_words, coll_likelihood_ratio, coll_pmi, terms_topia, vector_keybert, clean_bigrams, pyate_combo

path = "clusters"
shutil.rmtree(path, ignore_errors=True)
os.makedirs(path)

text_data = []

methods = {"indw":individual_words, "liker": coll_likelihood_ratio, "pmi": coll_pmi, "topia": terms_topia, "vbert": vector_keybert, "bigr": clean_bigrams, "combo": pyate_combo}

def txt_export(input_path):
    all_docs = []
    if os.path.isdir(input_path):
        all_docs = glob.glob(input_path + '/*.txt', recursive=True) + glob.glob(input_path + '/*.docx', recursive=True)
    elif os.path.splitext(input_path)[-1].lower() == ".txt" or os.path.splitext(input_path)[-1].lower() == ".docx":
        all_docs = [input_path]

    for file_path in all_docs:
        doc = os.path.basename(file_path).split('.')[0][:5]
        if os.path.splitext(file_path)[-1].lower() == ".txt":
            with open(file_path, encoding = "utf-8-sig") as file:
                text = file.read()
        elif os.path.splitext(file_path)[-1].lower() == ".docx":
            with open(file_path, "rb") as file:
                text = docx2python(file).text                       
       
        lines = re.sub("\n\n+", "\n", text).rstrip().splitlines()
                
        length = 0
        snip = ""
        count = 0
        for line in lines:
            line = re.sub("----\S+----", "", line)
            snip += line + "\n"
            length += len(line.split())
            if length >= args.piece:
                Btokenized = '\n'.join(text_to_sentences(text_to_words(par)) + '\n' for par in snip.splitlines()).strip()
                results = readability.getmeasures(Btokenized, lang='en')
                if results['sentence info']['words_per_sentence'] > args.words:
                    with open(os.path.join(path, f"{doc}_{str(count)}.txt"), "w", encoding="utf-8-sig") as f:
                        f.write(snip.rstrip())
                    text_data.append((f"{doc}_{str(count)}", results['readability grades']['FleschReadingEase'],
                                      results['readability grades']['LIX'],
                                      results['readability grades']['DaleChallIndex'],
                                      results['sentence info']['characters_per_word'],
                                      results['sentence info']['type_token_ratio'],                                      
                                      results['sentence info']['words_per_sentence'],
                                      results['sentence info']['words'],
                                      results['sentence info']['sentences'],
                                      results['sentence info']['paragraphs']
                                      ))
                length = 0
                snip = ""
                count += 1

            
def findOptimalEps(n_neighbors, data):
    '''
    function to find optimal eps distance when using DBSCAN; based on this article: https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
    '''
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', online=True, direction='increasing', interp_method='polynomial')
    knee.plot_knee()
    plt.show()

    print("Knee point", distances[knee.knee])
    return distances[knee.knee]

def text_clustering(data):
    global dbscan_labels
    
    if args.eps == None:
        eps = findOptimalEps(2, data)
    else:
        findOptimalEps(2, data)
        eps = args.eps

    dbscan = DBSCAN(eps=eps, min_samples=args.min_number, metric="euclidean")
    dbscan_labels = dbscan.fit_predict(data)
    df_dbscan = pd.DataFrame(data, columns = ['FleschReadingEase', 'LIX', 'DaleChallIndex', 'Char per word', 'Type token ratio'
                                              ])
    df_dbscan["Cluster"] = dbscan_labels

def cluster_visualization(data):
    # using PCA to reduce the dimenionality to two dimenions (to be able to plot the data with *seaborn*)
    plt.figure(figsize=(8,5))
    pca = PCA(n_components=2, whiten=False, random_state=42)
    data_pca = pca.fit_transform(data)
    df_dbscan_pca = pd.DataFrame(data=data_pca, columns=["pc_1", "pc_2"])
    df_dbscan_pca["Cluster"] = dbscan_labels
    sns_plot = sns.scatterplot(x="pc_1", y="pc_2", hue="Cluster", data=df_dbscan_pca)
    for i in range(df_dbscan_pca.shape[0]):
        plt.text(x=df_dbscan_pca.pc_1[i]+0.01, y=df_dbscan_pca.pc_2[i]+0.01, s=text_data[i][0], fontdict=dict(color='blue',size=8))
    plt.show()    

def docs_in_clusters(text_data_np):
    df_docs = pd.DataFrame(text_data_np[:, 0])
    df_docs["Cluster"] = dbscan_labels
    docs_clus = []
    for i in df_docs["Cluster"].unique():
        if i != -1:
            docs_clus.append((i, df_docs.query(f"Cluster == {i}")[0].values.tolist()))
    for item in docs_clus:
        dest = path + '/' + str(item[0])
        if os.path.isdir(dest) == False:
            os.makedirs(dest)
        for d in item[1]:
            shutil.move(path + "/" + d + ".txt", dest)            

def combinations_with_intersections(only_terms, method):
    selected_sets = []
    for u in range(len(only_terms), 2, -1):

        comb = combinations(only_terms, u)

        for big_comb in list(comb):
            good = True
            for sel in selected_sets:
                if set(map(frozenset, big_comb)).issubset(map(frozenset, sel)):
                    good = False
                    break
            if good:
                pair_comb = combinations(big_comb, 2)
                for i in list(pair_comb):
                    inters = i[0].intersection(i[1])
                    if method == "bigr":
                        # When using the bigram approach, the bigram lists may be of different length in different pieces,
                        # for intersection, Jaccard index is used 
                        if len(inters)/(len(i[0]) + len(i[1]) - len(inters)) > 0.015:
                            good = False
                            break                            
                    else:
                        if len(inters)/len(i[0]) > 0.2:
                            good = False
                            break
            if good:
                selected_sets.append(big_comb)          
    return selected_sets

def combinations_with_term_number(selected_term_sets, method):
    further_selected_sets = []
    for u in range(len(selected_term_sets), 2, -1):

        fcomb = combinations(selected_term_sets, u)

        for fbig_comb in list(fcomb):
            good = True
            x_fbig_comb, y_fbig_comb, z_fbig_comb = zip(*fbig_comb)
            for fsel in further_selected_sets:
                x_fsel, y_fsel, z_fsel = zip(*fsel)
                if frozenset(x_fbig_comb).issubset(frozenset(x_fsel)):
                    good = False
                    break
            if good:
                fpair_comb = combinations(fbig_comb, 2)
                for fi in list(fpair_comb):
                    if method == "bigr":
                        '''
                        For the bigram approach, we count the total number of terms for those that occur
                        more than 1 time. We take into account the possibility of dividing by 0
                        if there are no such terms. To do this, we add 1 in the denominator,
                        and set the limit to 2. The logic is as follows: if in the 1st piece 1 term
                        occurs 2 times, in the 2nd piece no terms occur 2 times or more, then
                        we still accept it: 2/(0+1) <= 2. If, for example, in the 1st piece 2 terms
                        occur 2 times, in the 2nd piece 1 term occurs 2 times, then we accept it: 4/(2+1) <= 2.
                        If in the 1st piece 2 terms occur 2 times, in the 2nd piece no terms
                        occur 2 times or more, then it’s bad: 4/(0+1) > 2
                        '''
                        if fi[0][2]/(fi[1][2] + 1) > 2 or fi[1][2]/(fi[0][2] + 1) > 2:
                            good = False
                            break                            
                    else:
                        if fi[0][2]/fi[1][2] > 1.3 or fi[1][2]/fi[0][2] > 1.3:
                            good = False
                            break
            if good:
                further_selected_sets.append(fbig_comb)          
    return further_selected_sets

def docs_without_common_terms():
    for unit in os.listdir(path):
        # We take for further processing by terms only folders (corresponding to clusters)
        # in which there are less files than the specified max_number limit (default: 10)
        # (otherwise, the search for combinations may be very long)
        if os.path.isdir(os.path.join(path, unit)):
            if len(os.listdir(os.path.join(path, unit))) > args.max_number:
                os.rename(os.path.join(path, unit), os.path.join(path, unit) + "-too big")
            else:
                test_sentences = []
                for f in os.listdir(os.path.join(path, unit)):
                    with open(os.path.join(path, unit, f), encoding="utf-8-sig") as ftfidf:
                        text_tfidf = ftfidf.read().lower()
                        test_sentences.append(text_tfidf)            

                for method in methods:
                    term_sets = []
                    only_terms = []
                    for doc, text in zip(os.listdir(os.path.join(path, unit)), test_sentences):
                        terms, total_count = methods[method](text)

                        term_sets.append((doc, terms, total_count))
                        only_terms.append(terms)
                        
                    selected_sets = combinations_with_intersections(only_terms, method)
                    num = 0
                   
                    
                    for selected_set in selected_sets:
                        selected_term_sets = []
                        for term_set in term_sets:
                            if term_set[1] in selected_set:
                                dest = os.path.join(path, unit + f"-{method}-{str(num)}")
                                if os.path.isdir(dest) == False:
                                    os.makedirs(dest)
                                shutil.copy(os.path.join(path, unit, term_set[0]), dest)
                                with open(os.path.join(path, unit + f"-{method}-{str(num)}", "term_file.txt"), "a", encoding="utf-8-sig") as term_file:
                                    term_file.write(term_set[0] + "\n" + str(term_set[1]) + "\nTotal_count " + str(term_set[2]) +
                                                    "\nFleishReadingEase\t" + str(round(float(text_data_np[text_data_np[:, 0] == term_set[0].split('.')[0], 1][0]), 2)) +
                                                    "\tLIX\t" + str(round(float(text_data_np[text_data_np[:, 0] == term_set[0].split('.')[0], 2][0]), 2)) +
                                                    "\tDaleChallIndex\t" + str(round(float(text_data_np[text_data_np[:, 0] == term_set[0].split('.')[0], 3][0]), 2)) +
                                                    "\tChar per word\t" + str(round(float(text_data_np[text_data_np[:, 0] == term_set[0].split('.')[0], 4][0]), 2)) +
                                                    "\tType token ratio\t" + str(round(float(text_data_np[text_data_np[:, 0] == term_set[0].split('.')[0], 5][0]), 2)) +
                                                    "\tWORDS PER SENT\t" + str(round(float(text_data_np[text_data_np[:, 0] == term_set[0].split('.')[0], 6][0]), 2)) +
                                                    "\twords\t" + str(text_data_np[text_data_np[:, 0] == term_set[0].split('.')[0], 7][0]) +
                                                    "\tsentences\t" + str(text_data_np[text_data_np[:, 0] == term_set[0].split('.')[0], 8][0]) +
                                                    "\tparagraphs\t" + str(text_data_np[text_data_np[:, 0] == term_set[0].split('.')[0], 9][0]) + 
                                                    "\n")
                                
                                selected_term_sets.append(term_set)

                        # Here we  can do combinations from the sets "document - terms - number of terms"
                        further_selected_sets = combinations_with_term_number(selected_term_sets, method)
                        fnum = 0  
                        for further_selected_set in further_selected_sets:
                            for term_set in term_sets:
                                if term_set in further_selected_set:
                                    fdest = os.path.join(dest + f"-{str(fnum)}")
                                    if os.path.isdir(fdest) == False:
                                        os.makedirs(fdest)
                                    shutil.copy(os.path.join(path, unit, term_set[0]), fdest)
                            fnum += 1

                        num += 1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Clustering of homogeneous text pieces',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', type=str,
                        help='input folder or file. The allowed extensions are .txt and .docx', required=True)

    parser.add_argument('-p', '--piece', type=int,
                        help='targeted word count of text pieces for clustering', default=500)

    parser.add_argument('-w', '--words', type=int,
                        help='minimum average word count per sentence in text pieces for clustering', default=6)

    parser.add_argument('-e', '--eps', type=float,
                        help='eps distance for DBSCAN')
    
    parser.add_argument('-m', '--min_number', type=int,
                        help='minimum number of pieces in cluster', default = 3)    
    
    parser.add_argument('-x', '--max_number', type=int,
                        help='maximum number of pieces in cluster', default = 10)
    
    parser.add_argument('-s', '--scaler', type=str, choices = ["min-max", "z-score"],
                        help='min-max or z-score scaler for normalization. Default: min-max', default = "min-max")

    args = parser.parse_args()

    txt_export(args.input)
    
    text_data_np = np.array(text_data)
    
    if args.scaler == "min-max":
        scaler = MMS()
    else:
        scaler = SS()

    data = scaler.fit_transform(text_data_np[:, 1:6])
    
    text_clustering(data)
    
    cluster_visualization(data)
    
    docs_in_clusters(text_data_np)

    docs_without_common_terms()

    
    
