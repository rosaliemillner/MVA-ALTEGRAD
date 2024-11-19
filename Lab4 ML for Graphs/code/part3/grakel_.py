import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')


def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab) #add word in vocab 

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab #we end up only with the words we have encountered in the docs


path_to_train_set = '../datasets/train_5500_coarse.label'
path_to_test_set = '../datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab)) #7151


import networkx as nx
import matplotlib.pyplot as plt

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    
    for idx,doc in enumerate(docs):
        G = nx.Graph()
        for i, word1 in enumerate(doc):
            G.add_node(vocab[word1])
            for j in range(i + 1, min(i + window_size, len(doc))):
                word2 = doc[j]
                if not G.has_edge(word1, word2):
                    G.add_edge(vocab[word1], vocab[word2])

        graphs.append(G)
    
    return graphs

vocab_dict = {v:k for k,v in vocab.items()}

# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
for G in G_train_nx: nx.set_node_attributes(G, vocab_dict, 'label')

G_test_nx = create_graphs_of_words(test_data, vocab, 3)
for G in G_test_nx: nx.set_node_attributes(G, vocab_dict, 'label')

print("Example of graph-of-words representation of document") ######
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Task 12

#Transform networkx graphs to grakel representations
# G_train = [graph_from_networkx(g, node_labels_tag='label') for g in list(G_train_nx)] # your code here #
# G_test = [graph_from_networkx(g, node_labels_tag='label') for g in list(G_test_nx)] # your code here #
G_train = graph_from_networkx(G_train_nx, node_labels_tag = 'label')
G_test = graph_from_networkx(G_test_nx, node_labels_tag = 'label')

# print(type(G_train[0]))  #<class 'networkx.classes.graph.Graph'>
# print(type(G_test[0]))

#Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True) # your code here #

#Construct kernel matrices
K_train = gk.fit_transform(G_train) # your code here #
K_test = gk.transform(G_test)# your code here #

#Task 13

# Train an SVM classifier and make predictions

svm = SVC(kernel='precomputed')

svm.fit(K_train, y_train)
y_pred = svm.predict(K_test)

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred, y_test))


#Task 14

# import grakel
# print(dir(grakel.kernels))

from grakel.kernels import RandomWalkLabeled, NeighborhoodHash

G_train = graph_from_networkx(G_train_nx, node_labels_tag = 'label')
G_test = graph_from_networkx(G_test_nx, node_labels_tag = 'label')

gk_rw = RandomWalkLabeled()
K_train_rw = gk_rw.fit_transform(G_train)
K_test_rw = gk_rw.transform(G_test)

clf_rw = SVC(kernel="precomputed")
clf_rw.fit(K_train_rw, y_train)
y_pred_rw = clf_rw.predict(K_test_rw)

print("Accuracy for RandomWalkLabeled:", accuracy_score(y_pred_rw, y_test))


gk_nh = NeighborhoodHash()
K_train_nh = gk_nh.fit_transform(G_train)
K_test_nh = gk_nh.transform(G_test)

clf_nh = SVC(kernel="precomputed")
clf_nh.fit(K_train_nh, y_train)
y_pred_nh = clf_rw.predict(K_test_nh)

print("Accuracy for NeighborhoodHashLabeled:", accuracy_score(y_pred_rw, y_test))

#the computation is costly, so I will not be comparing many different kernels