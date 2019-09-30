from SPARQLWrapper import SPARQLWrapper, JSON
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from operator import itemgetter
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix
import itertools
from scipy.spatial.distance import cdist


# returns the fraction of vector that has in their neighborhood the correct labelled point
# returns also the predicted_label for each vector prediction

# mode = both: model has to be a list [t2v_model, hyper_model]
               # returns a dict
#      = hyper: model has to be hyper_model
#      = t2v: model has to be t2v_model

# def check_prediction(labels, vectors, embedding, mode, topn=10):
#     predicted_labels = ['a' for a in labels]
#     acc1 = -1
#     acc2 = -1
#     if mode == 'both':
#         acc1, lab1 = check_prediction(labels, vectors[1], embedding[1], 'hyper', 751)
#         acc2, lab2 = check_prediction(labels, vectors[0], embedding[0], 't2v', 403)
#         return {'hyp' : acc1, 't2v' : acc2, 'hypLabel' : lab1, 't2vLabel' : lab2}
#     else:
#         if topn == 'all' and mode == 'hyper':
#             topn = 751
#         elif topn == 'all' and mode == 't2v':
#             topn = 403
#         correct = 0
#         all_labels = list(set(labels))
#         for i, (label, vector) in enumerate(zip(labels, vectors)):
#             sim, neigh = find_neighbours(vector=vector, 
#                                                model=embedding, 
#                                                mode=mode,
#                                                topn=topn)
            
#             try:
#                 predicted_labels[i] = neigh[np.min(np.where(np.isin(neigh, all_labels)))]
#             except:
#                 predicted_labels[i] = 'Unrecognized'
#             if label == predicted_labels[i]:
#                 correct += 1
                
#         return correct/len(labels), predicted_labels
    
# def check_prediction(labels, vectors, embedding, mode, parenthood, topn, tensors = 0):
#     if mode == 'both':
#         tensors = [v for v in embedding[1].values()]
#         lab2, err2 = check_prediction(labels, vectors[0], embedding[0], 't2v', parenthood, topn)
#         lab1, err1 = check_prediction(labels, vectors[1], embedding[1], 'hyper', parenthood, topn, tensors)
#         return {'hypLabel' : lab1, 't2vLabel' : lab2, 'hypErr': err1, 't2vErr': err2}
#     else:
#         if topn == 'all' and mode == 'hyper':
#             topn = 751
#         elif topn == 'all' and mode == 't2v':
#             topn = 403
#         correct = 0
#         all_labels = list(set(labels))
        
        
#         predicted_labels = ['a' for a in labels]
        
#         pred_errors = {label: [] for label in all_labels}
#         nan_counter = 0
        
#         tot = len(labels)
#         prec = 0
        
#         for i, (label, vector) in enumerate(zip(labels, vectors)):
#             sim, neigh = find_neighbours(vector=vector, 
#                                                model=embedding, 
#                                                mode=mode,
#                                                tensors=tensors,
#                                                topn=topn)
#             if not np.isnan(sim[0]): 
#                 try:
#     #                 if i < 15:
#                     pos = np.min(np.where(np.isin(neigh, all_labels)))
#                     predicted_labels[i] = neigh[pos]
# #                     if mode == 'hyper':
#     #                     print('----------------------')
#     #                     print('neigh: {}'.format(neigh[:10]))
#     #                     print('pos: {}'.format(pos))
#     #                     print('label: {}, pred_label : {}'.format(label, predicted_labels[i]))
#                 except:
#                     print('unrecognized, i: {}'.format(i))
#                     predicted_labels[i] = 'Unrecognized'

#     #             print('---------------------------------')
#     #             print('label: {}\n ranking: {},\n pred_lab: {}'.format(label, list(zip(neigh, sim))[:10], predicted_labels[i]))

#     #           Mi concentro sugli errori
#                 j = 0
#                 flag = True
#                 while flag and j < pos:
#                     if neigh[pos] != label: 
#                         n = neigh[j]
#                         if n in parenthood[label]['ancestors']:
#                             pred_errors[label].append('ancestors')
#                             flag = False    

#                         elif n in parenthood[label]['sons']:
#                             pred_errors[label].append('sons')
#                             flag = False                    

#                         elif n in parenthood[label]['siblings']:
#                             pred_errors[label].append('siblings')
#                             flag = False
#                     j += 1

#                 if flag and neigh[pos] != label:
#                     pred_errors[label].append('errors')
#                 if flag and neigh[pos] == label:
#                     pred_errors[label].append('correct')
#             else:
#                 nan_counter += 1
                
#             perc = round(i/tot, 2)
            
#             if perc != prec:
#                 prec = perc
#                 clear_output()
#                 print(perc)
        
#         print('nan: {}'.format(nan_counter))
#         return predicted_labels, pred_errors
# returns cosine similarity between two vectors
def cos_sim(v1, v2):
    return np.dot(v1,v2)/(norm(v1) * norm(v2))

# returns the 'topn' neighbours of vector 'node' in the embedding "model", 
#   use hyperbolic distance if mode is hyper and model is nickel's hyperbolic embedding
#   use cosine similarity if mode is t2v and model is Gensim's word2vec model about Type (type2vec)

# def double_plot(mo_mixed_model, testset, iteration, frac, classes, parenthood):
    
#     embeddings = {'t2v': type_embedding, 'hyper': hyperbolic_embedding}
#     mo_y_predict = mo_mixed_model.predict(testset[:frac])
# #     mo_y_predict[1] = xscaler.inverse_transform(mo_y_predict[1])
#     res = check_prediction(labels=Y_test[:frac], 
#                        vectors=mo_y_predict,
#                        embedding=[embeddings['t2v'], embeddings['hyper']],
#                        topn='all',
#                        parenthood=parenthood,
#                        mode='both')
    
#     plt.figure(figsize=(40, 40))
#     pred = res['hypLabel']
#     plt.subplot(221)
#     cnf_matrix = confusion_matrix(y_true = Y_test[:frac], y_pred = pred)
#     plot_confusion_matrix(cnf_matrix, classes = classes, title='Hyperbolic middle out {}'.format(iteration))
    
#     acc = accuracy_score(y_true=Y_test[:frac], y_pred=pred)
#     prec = precision_score(y_true=Y_test[:frac], y_pred=pred, average='weighted')
#     rec = recall_score(y_true=Y_test[:frac], y_pred=pred, average='weighted')
#     f1 = f1_score(y_true=Y_test[:frac], y_pred=pred, average='weighted')

#     print('Accuracy:{} Precision:{} Recall:{} F1:{}'.format(round(acc, 3),
#                                                             round(prec, 3), 
#                                                             round(rec, 3), 
#                                                             round(f1, 3)))

#     plt.subplot(222)
#     error_matrix(n_class = len(classes), dic=res['hypErr'], classes=classes)

#     plt.subplot(223)
#     pred = res['t2vLabel']
#     cnf_matrix = confusion_matrix(y_true = Y_test[:frac], y_pred = pred)
#     plot_confusion_matrix(cnf_matrix, classes = classes, title='t2v middle out {}'.format(iteration))

#     acc = accuracy_score(y_true=Y_test[:frac], y_pred=pred)
#     prec = precision_score(y_true=Y_test[:frac], y_pred=pred, average='weighted')
#     rec = recall_score(y_true=Y_test[:frac], y_pred=pred, average='weighted')
#     f1 = f1_score(y_true=Y_test[:frac], y_pred=pred, average='weighted')

#     print('Accuracy:{} Precision:{} Recall:{} F1:{}'.format(round(acc, 3),
#                                                             round(prec, 3), 
#                                                             round(rec, 3), 
#                                                             round(f1, 3)))


    
#     plt.subplot(224)
#     error_matrix(n_class = len(classes), dic=res['t2vErr'], classes=classes)

def find_neighbours(vector, model, mode, tensors, topn=10):
    node_distances = []
    if mode=='hyper':
        tensors = [v.numpy() for v in list(model['model'].items())[0][1]]
        d = cdist([vector], tensors, hyper_distance)
        node_distances = [[label, distance] for label, distance in zip(model['objects'], d[0])]
        
        sorted_list = sorted(node_distances, key=itemgetter(1))
    elif mode=='t2v':
        sorted_list = model.similar_by_vector(vector, topn=topn, restrict_vocab=None)
    similarities = [x[1] for x in sorted_list]
    labels = [x[0] for x in sorted_list]
    if topn == 'all':
        return similarities, labels
    else:
        return similarities[:topn], labels[:topn]

# def error_matrix(n_class, dic, classes):
#     error_matrix_dict = {'correct' : [0 for x in range(n_class)],
#                          'sons' : [0 for x in range(n_class)],
#                          'ancestors': [0 for x in range(n_class)],
#                          'siblings': [0 for x in range(n_class)],
#                          'errors': [0 for x in range(n_class)]}

#     for i, clas in enumerate(classes):
#         counter = Counter(dic[clas])
#         for k in error_matrix_dict.keys():
#     #         print(k)
#             error_matrix_dict[k][i] = round(counter[k] / sum(counter.values()), 2)

#     error_pd = pd.DataFrame(data=error_matrix_dict)

#     sns.heatmap(error_pd, annot=True, cmap="YlGnBu", yticklabels=classes)

# def find_neighbours(vector, model, mode, tensors, topn=10):
#     node_distances = []
#     if mode=='hyper':
#         d = cdist([vector], tensors, hyper_distance)
        
#         node_distances = [(label, distance) for label, distance in zip(model.keys(), d[0])]
#         if np.isnan(node_distances[0][1]):
#             print('norm: {}'.format(norm(vector)))
            
#         sorted_list = sorted(node_distances, key=itemgetter(1))
#     elif mode=='t2v':
#         sorted_list = model.similar_by_vector(vector, topn=topn, restrict_vocab=None)
#     similarities = [x[1] for x in sorted_list]
#     labels = [x[0] for x in sorted_list]
#     if topn == 'all':
#         return similarities, labels
#     else:
#         return similarities[:topn], labels[:topn]

    # returns datasets for supervised learning

# data has to be a dict: {'class' : [list of words]}
# word_embedding: gensim's Word2Vec model that contains every word in data
# hyper_embedding: Nickel's hyperbolic embedding, contains a list of labelled vector, contain every class in data
# type_embedding: gensim's Word2Vec model that contains every class in data 

def get_datasets(data, word_embedding, hyper_embedding, type_embedding, test_size):
    X = []
    Y_type = []
    Y_hyperbolic = []
    labels = []
    for key in data.keys():
        for word in data[key]:
            X.append(word_embedding.wv[word])
            Y_type.append(type_embedding.wv[key])
            Y_hyperbolic.append(get_hyperbolic_vector(key, hyper_embedding))
            labels.append(key)
    
    Y_hyperbolic = tensor_to_vector(Y_hyperbolic)

    Y = [[t, h, l] for t, h, l in zip(Y_type, Y_hyperbolic, labels)]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    
    y_t2v_train = np.array([y[0] for y in y_train])
    y_t2v_test = np.array([y[0] for y in y_test])
    y_hyp_train = np.array([y[1] for y in y_train])
    y_hyp_test = np.array([y[1] for y in y_test])
    labels_train = [y[2] for y in y_train]
    labels_test = [y[2] for y in y_test]
    
    
    print('{} train input \n{} t2v_train_output\n{} hyper_train_output'.format(len(X_train), 
                                                                             len(y_t2v_train), 
                                                                             len(y_hyp_train)))
    print('{} test input \n{} t2v_test_output\n{} hyper_test_output'.format(len(X_test), 
                                                                          len(y_t2v_test), 
                                                                          len(y_hyp_test)))    
    
    return np.array(X_train), np.array(X_test), y_hyp_train, y_hyp_test, y_t2v_train, y_t2v_test, labels_train, labels_test

# returns dbpedia's resources with type=classtype
# returns a list of strings
def get_from_class(classtype):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("""
    PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX vrank:<http://purl.org/voc/vrank#>
    PREFIX dbo:<http://dbpedia.org/ontology/>
        SELECT distinct ?s ?v
        FROM <http://dbpedia.org> 
        FROM <http://people.aifb.kit.edu/ath/#DBpedia_PageRank> 
        WHERE { ?s rdf:type """ + classtype + """.
                ?s vrank:hasRank/vrank:rankValue ?v.
               }  ORDER BY DESC(?v)
    """)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    collector = []

    for result in results["results"]["bindings"]:
        data_a = result["s"]["value"].replace("http://dbpedia.org/resource/", "")
        collector.append((data_a))
    return collector

# returns hyperbolic vector about Type in model (hyperbolic embedding of types) 
def get_hyperbolic_vector(Type, model):
    points = model['model']['lt.weight']
    objects = model['objects']
    
    return points[objects.index(Type)]


def get_parenthood(classes):
    parenthood = {}
    
    for clas in classes:
        
        parenthood[clas] = {}
        
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery("""
        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX vrank:<http://purl.org/voc/vrank#>
        PREFIX dbo:<http://dbpedia.org/ontology/>
            SELECT distinct ?s
            FROM <http://dbpedia.org> 
            FROM <http://people.aifb.kit.edu/ath/#DBpedia_PageRank> 
            WHERE { ?s rdfs:subClassOf dbo:""" + clas + """.
                   }
        """)

        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        collector = []

        for result in results["results"]["bindings"]:
            data_a = result["s"]["value"].replace("http://dbpedia.org/ontology/", "")
            if 'http://www.w3.org/2002/07/owl#Thing' not in data_a:
                collector.append((data_a))
        
        parenthood[clas]['sons'] = collector
        
        parenthood[clas]['ancestors'] = []
        ancestor_flag = True
        q = [clas]
        i = 0
        while i<len(q):
            node = q[i]
            sparql.setQuery("""
            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX vrank:<http://purl.org/voc/vrank#>
            PREFIX dbo:<http://dbpedia.org/ontology/>
                SELECT distinct ?s
                FROM <http://dbpedia.org> 
                FROM <http://people.aifb.kit.edu/ath/#DBpedia_PageRank> 
                WHERE { dbo:""" + node + """ rdfs:subClassOf ?s.
                       }
            """)

            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            collector = []

            for result in results["results"]["bindings"]:
                data_a = result["s"]["value"].replace("http://dbpedia.org/ontology/", "")
                data_a = data_a.replace("http://schema.org/", "")
                if 'http://www.w3.org/2002/07/owl#Thing' not in data_a:
                    collector.append((data_a))
                    if data_a not in q:
                        q.append(data_a)
                else:
                    ancestor_flag = False
            
            parenthood[clas]['ancestors'].extend(collector)
            parenthood[clas]['ancestors'] = list(set(parenthood[clas]['ancestors']))
            i += 1
        
        
        sparql.setQuery("""
        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX vrank:<http://purl.org/voc/vrank#>
        PREFIX dbo:<http://dbpedia.org/ontology/>
            SELECT distinct ?v ?s
            FROM <http://dbpedia.org> 
            FROM <http://people.aifb.kit.edu/ath/#DBpedia_PageRank> 
            WHERE { dbo:""" + clas + """ rdfs:subClassOf ?s.
                    ?v rdfs:subClassOf ?s.
                   }
        """)

        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        collector = []

        for result in results["results"]["bindings"]:
            data_a = result["v"]["value"].replace("http://dbpedia.org/ontology/", "")
            data_b = result["s"]["value"].replace("http://dbpedia.org/ontology/", "")
            if 'http://www.w3.org/2002/07/owl#Thing' not in data_b:
                collector.append((data_a))
        parenthood[clas]['siblings'] = [c for c in collector if c != clas]        
        
    return parenthood

# returns the hyperbolic distance between 2 vectors 
def hyper_distance(tensor1, tensor2, r=1):
    return np.arccosh(
        1 + ((2 * r**2 * (norm(tensor1 - tensor2) ** 2))
             / ((r**2 - norm(tensor1) ** 2)*(r**2 - norm(tensor2) ** 2))))

# plot 4 confusion matrices, based on 4 prediction (hyper, t2v, mixed_hyper, mixed_t2v) 
# of X_test using models: hyper_model, t2v_model and mixed_models
# calls 4 time the function plot_confusion_matrix
# def plot_all(X_test, labels_test, models, embeddings, types, topn):
    
#     hyper_model = models['hyper']
#     y_hyper_predict = hyper_model.predict(X_test)
#     hyper_acc, hyper_pred= check_prediction(labels=labels_test,
#                                             vectors=y_hyper_predict, 
#                                             embedding=embeddings['hyper'], 
#                                             topn=topn,
#                                             mode='hyper')
    
#     print('30%')
    
#     t2v_model = models['t2v']
#     y_t2v_predict = t2v_model.predict(X_test)
#     t2vacc, t2v_pred = check_prediction(labels=labels_test, 
#                                     vectors=y_t2v_predict,
#                                     embedding=embeddings['t2v'], 
#                                     topn=topn, 
#                                     mode='t2v')
#     clear_output()
#     print('60%')
    
#     mixed_model = models['mixed']    
#     y_predict = mixed_model.predict(X_test)
#     res = check_prediction(labels=labels_test, 
#                        vectors=y_predict,
#                        embedding=[embeddings['t2v'], embeddings['hyper']],
#                        topn=topn,
#                        mode='both')
#     clear_output()
#     print('90%')  
    
    
#     classes = [x.replace('dbo:', '') for x in types]
#     classes = classes + ['Unrecognized']
#     classes = sorted(classes)

#     plt.figure(figsize=(15, 15))
    
#     plt.subplot(221)
#     cnf_matrix = confusion_matrix(y_true = labels_test, y_pred = hyper_pred)
#     plot_confusion_matrix(cnf_matrix, classes = classes, title='Hyperbolic')
    
    
#     plt.subplot(222)
#     pred = res['hypLabel']
#     cnf_matrix = confusion_matrix(y_true = labels_test, y_pred = pred)
#     plot_confusion_matrix(cnf_matrix, classes = classes, title='Hyperbolic on mixed')
    
    
#     plt.subplot(223)
#     cnf_matrix = confusion_matrix(y_true = labels_test, y_pred = t2v_pred)
#     plot_confusion_matrix(cnf_matrix, classes = classes, title='T2V')
    
    
#     plt.subplot(224)
#     pred = res['t2vLabel']
#     cnf_matrix = confusion_matrix(y_true = labels_test, y_pred = pred)
#     plot_confusion_matrix(cnf_matrix, classes = classes, title='T2V on mixed')
    
#     plt.savefig('Hits@{}'.format(topn))
    

# plot the confusion matrix cm and the classes 'classes' ordered in alfabetical order
# def plot_confusion_matrix(cm, classes,
#                           normalize=True,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
    
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.savefig('./fig.png')

# plot the loss about a network history passed as parameter
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# cast a list of tf tensor in an array of numpy array
def tensor_to_vector(Y_hyperbolic):
    return np.array(list(map(lambda x : x.numpy(),Y_hyperbolic)))

# Functions to find the hyperbolic midpoint
def lorentzian(v0, v1):
    prod = 0
    for i in range(len(v0) - 1):
        prod += v0[i] * v1[i]
    
    ret = - prod + (v0[len(v0) - 1] * v1[len(v1) - 1])
    return abs(ret)


def hyperboloid_projection(v, r):
    n = norm(v)
    t = [(r**2 + (n ** 2)) / (r**2 - (n ** 2))]
    projected = [(2 * r**2 * vs) /(r**2 - (n ** 2)) for vs in v]
    projected.extend(t)
    return np.array(projected)

def inverse_projection(v, r):
    return np.array([vs/(r**2 + v[-1]) for vs in v[:-1]])

def midpoint(vectors, r = 1, n = False):
    if n:
        hv = [hyperboloid_projection(vs, r) if norm(vs) < 1 else - hyperboloid_projection(vs, r) for vs in vectors]
    else:
        hv = [hyperboloid_projection(vs, r) for vs in vectors]
    summ = sum(hv)
    midpoint = summ/np.sqrt(lorentzian(summ, summ))
    poincarè_midpoint = inverse_projection(midpoint, r)
    
    return poincarè_midpoint 