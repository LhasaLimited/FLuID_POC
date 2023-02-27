"""
    Cronos library
    This is the main python library for the Cronos proof of concept project
"""
import os
import random
import time
from os import path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from matplotlib._cm_listed import cmaps
import matplotlib.pyplot as plt
from plotly.graph_objs import *
from plotly.subplots import make_subplots
import seaborn as sns

from rdkit import DataStructs, Chem
from rdkit.Chem import PandasTools, AllChem
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from xgboost import XGBClassifier
from IPython.display import display, HTML
from tqdm import tqdm_notebook as tqdm

pd.options.mode.chained_assignment = None  # default='warn'

"""
   Displays Plotly plots 
---
  plot : the plot to show
  renderer : the renderer to use
  pause : if true makes a pause to avoid Ploty's image numbering bug
"""

def showPlot(plot, renderer=None, pause=False):
    if renderer == None : 
        plot.show() 
    else :
        plot.show(renderer = renderer)
        
    # plotly figure numbering workaround
    if(pause) : time.sleep(1)
    
"""
    Computes the fingerprint for a given molecule
---
    molecule : the molecule to fingerprint
"""

def compute_fp_array(molecule):
   #arr = pd.array(0)
    arr = np.zeros((0,), dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048, useFeatures=True)
    DataStructs.ConvertToNumpyArray(fp, arr)
    #arr = arr.astype(pd.uint8)
    return arr


"""
    Creates a classifier
---
    algorithm : the algorithm to use for the classifier
"""

def create_classifier(algorithm):
    if algorithm == 'rf':
        return RandomForestClassifier(n_estimators=150, max_depth=15, class_weight='balanced', random_state=0)
    elif algorithm == 'nb':
        return GaussianNB()
    elif algorithm == 'svm':
        return SVC()
    elif algorithm == 'svml':
        return SVC(kernel="linear", C=0.025)
    elif algorithm == 'xgb':
        return XGBClassifier(objective="binary:logistic", random_state=42)
    elif algorithm == 'knn':
        return KNeighborsClassifier(8)
    elif algorithm == 'gp':
        return GaussianProcessClassifier(1.0 * RBF(1.0))
    elif algorithm == 'dt':
        return DecisionTreeClassifier(max_depth=5)
    elif algorithm == 'mlp':
        return MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(40, 40))
    elif algorithm == 'ab':
        return AdaBoostClassifier()
    else:
        raise ValueError('"{foo}" wrong algorithm'.format(foo=algorithm))


"""
    Gets the list of supported and explorable algorithms
"""

def get_algorithms():
    return ['rf', 'nb', 'svm', 'xgb', 'knn', 'dt', 'mlp']


"""
    Create and train a classifier using a given algorithm and training data
---
    algorithm : the algorithm to use for the classifier
         data : the data to use to train classifier
"""

def create_trained_classifier(algorithm, data, balance=True):
    X = np.asarray([fp for fp in data.FP])
    Y = np.asarray([c for c in data.CLASS])

   
    if balance :
        # RUS
        rus = RandomUnderSampler(random_state=0)
        X_resampled, Y_resampled = rus.fit_resample(X, Y)
        # ROS
        #ros = RandomOverSampler(random_state=0)
        #X_resampled, Y_resampled = ros.fit_resample(X, Y)
        # SMOTE
        #X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
    else :
        #No balancing
        X_resampled, Y_resampled = X,Y
        
    
    print("Classifier building Balanced=" + str(balance) + " data size " + str(len(data)) + " sample size : " + str(len(X_resampled)))
    classifier = create_classifier(algorithm)
    classifier.fit(X_resampled, Y_resampled)
    return classifier


"""
    Creates a new validation result table with the desired columns
"""

def create_validation_table():
    table = pd.DataFrame(
        columns=['Model', 'Size', 'BAC', 'ACC', 'MCC', 'MCC-STD', 'F1', 'SENS', 'SPEC', 'PPV', 'NPV', 'AUC'])
    return table


"""
    Adds a validation row to a validation table
---
     table : the table to append
    y_true : the true label values
    y_pred : the predicted label values
     title : the title for this row
      size : the size of the training set used when producing this row
"""

def compute_validation_row(table, y_true, y_pred, title, size):
    row = pd.Series([title, size,
                                    metrics.balanced_accuracy_score(y_true, y_pred),
                                    metrics.accuracy_score(y_true, y_pred),
                                    metrics.matthews_corrcoef(y_true, y_pred), 0,
                                    metrics.f1_score(y_true, y_pred),
                                    metrics.recall_score(y_true, y_pred, average='macro', labels=[1]),
                                    metrics.recall_score(y_true, y_pred, average='macro', labels=[0]),
                                    metrics.precision_score(y_true, y_pred, average='macro', labels=[1]),
                                    metrics.precision_score(y_true, y_pred, average='macro', labels=[0]),
                                    metrics.roc_auc_score(y_true, y_pred, average='macro', labels=[1]),
                                    ], index=table.columns)
    return row


"""
     Adds a validation row given true and predicted values
---
         table : the table to append
        y_true : the true labels
        y_pred : the predicted labels
         title : the title of this row (informative)
          size : the size of the training (informative)
"""
def add_validation_row(table, y_true, y_pred, title, size):
    row = compute_validation_row(table, y_true, y_pred, title, size)
    table = table.append(row, ignore_index=True)
    return table

"""
     Adds a validation row given true and predicted values
---
         table : the table to append
    classifier : the claissifier to validate
     test_data : the test data
    model name : the name of the model (informative)
          size : the size of the training (informative)
"""
def create_validation_row(table, classifier, test_data, model_name, size):
    X = np.asarray([fp for fp in test_data.FP])
    Y_true = np.asarray([c for c in test_data.CLASS])
    Y_pred = classifier.predict(X)
    row = compute_validation_row(table, Y_true, Y_pred, model_name, size)
    return row

"""
    Validates a given classifier and adds the results
    to a given validation result table
---
         table : the table to append
    classifier : the classifier to validate
     test_data : the test data to use
    model_name : the name of the model setup for this row
          size : the size of the training set used when producing this row
"""

def add_classifier_validation(table, classifier, test_data, model_name, size):
    X = np.asarray([fp for fp in test_data.FP])
    Y_true = np.asarray([c for c in test_data.CLASS])
    Y_pred = classifier.predict(X)
    table = add_validation_row(table, Y_true, Y_pred, model_name, size)
    return table


"""
    Adds a validation with standard deviation for MCC extracted from 
    a table of replicated validations for a given model
---
          table : the table to append
replicate_table : the classifier to validate
     model_name : the title for this row
           size : the size of the training set used when producing this row
"""

def add_replicate_table_row(table, replicate_table, model_name, column='Model'):
    # add an entry in the table for the corresponding model
    table = table.append(pd.Series([model_name], index=[column])
                         .append(replicate_table[replicate_table[column] == model_name].mean(axis=0)), ignore_index=True)

    # compute standard and 95% confidence interval deviation in replicates
    std = replicate_table[replicate_table[column] == model_name]['MCC'].std()
    c95 = std * 1.96
    index = len(table.index) - 1

    # update columns MCC-C95 and MCC-STD
    if index == 0:
        table['MCC-C95'] = c95
    else:
        table['MCC-C95'][index] = c95
    if index == 0:
        table['MCC-STD'] = std
    else:
        table['MCC-STD'][index] = std
    return table

    

"""
    5 times cross-validates a given classifier and adds the results
    to a given validation result table
---
         table : the table to append
    classifier : the classifier to validate
          data : the test data to use
         title : the title for this row
"""

def add_classifier_cross_validation(table, classifier, data, title):
    X = np.asarray([fp for fp in data.FP])
    y_true = np.asarray([c for c in data.CLASS])
    y_pred = cross_val_predict(classifier, X, y_true, cv=5)
    table = add_validation_row(table, y_true, y_pred, title, int(len(data.index) * 4 / 5))
    return table


"""
    Appends cluster data to a table
---
      table : the table to append
       data : the data to add
        tag : the cluster index
       size : the total size of the origin data
"""

def add_cluster_data(table, name, prefix, data, tag, size):
    inactive = data[data.ACTIVITY == "Inactive"].ACTIVITY.count()
    active = data[data.ACTIVITY == "Active"].ACTIVITY.count()
    total = inactive + active
    proportion = total / size * 100
    return table.append({name: prefix + str(tag), 'Active': active, 'Inactive': inactive, 'Total': total,
                         'Proportion (%)': proportion}, ignore_index=True)


"""
    Samples a subset of a dataset
---
      data: the dataset to sample
     limit: the maximum number of instances to sample
---
    sample: the sample of the dataset
"""

def cap(data, limit=500):
    return data.sample(min(len(data),limit))


"""
    Loads the training data
---
      params: the experiment configuration
      force : if true forces to reload from SD file

"""

def load_training_data(params, force=True):
    training_data_file = params['training_data_file']
    pickleFile = os.path.join("data", training_data_file + ".pkl")

    # use pre-converted pickle file if force = false
    # else reconvert the training data from the SDF file
    if not force and path.exists(pickleFile):
        data = pd.read_pickle(pickleFile)
        print("From pickle training data size = " + str(data.shape[0]))
    else:
        sdFile = os.path.join("data", training_data_file + ".sdf")
        data = PandasTools.LoadSDF(sdFile, molColName='MOLECULE')
        data['INCHI'] = [Chem.inchi.InchiToInchiKey(Chem.inchi.MolToInchi(mol)) for mol in data.MOLECULE]
        data.rename({'Lhasa Chembl Call': 'ACTIVITY'}, axis=1, inplace=True)
        data['CLASS'] = [(0 if cls == 'Inactive' else 1) for cls in data.ACTIVITY]
        data['FP'] = data['MOLECULE'].apply(compute_fp_array)
        data['SOURCE'] = training_data_file
        data['ROLE'] = 'Training'
        data.drop(['inchi_key', 'compound_id', 'assay_id', 'ID'], axis=1, inplace=True)

        # Remove internal duplicates
        print("From SDF training data size = " + str(data.shape[0]))
        data.drop_duplicates(subset='INCHI', keep='first', inplace=True)
        print("   -After removing dupplicates size = " + str(data.shape[0]))
        data.to_pickle(pickleFile)

    return data


"""
    Loads the test data (Preissner)
---
      test_data_file : the file to load
      training_data : the training data (used to remove duplicates)
      force : if true forces to reload from SD file

"""

def load_test_data(training_data, params, force=True):
    pickleFile = os.path.join("data", params['test_data_file'] + ".pkl")

    # use pre-converted pickle file if force = false
    # else reconvert the training data from the SDF file
    if not force and path.exists(pickleFile):
        data = pd.read_pickle(pickleFile)
        print("From pickle test data size = " + str(data.shape[0]))
    else:
        sdFile = os.path.join("data", params['test_data_file'] + ".sdf")
        data = PandasTools.LoadSDF(sdFile, molColName='MOLECULE')
        data['INCHI'] = [Chem.inchi.InchiToInchiKey(Chem.inchi.MolToInchi(mol)) for mol in data.MOLECULE]
        data.rename({'Overall Conservative Call': 'ACTIVITY'}, axis=1, inplace=True)
        data['CLASS'] = [(0 if cls == 'Inactive' else 1) for cls in data.ACTIVITY]
        data['FP'] = data['MOLECULE'].apply(compute_fp_array)
        data['SOURCE'] = params['test_data_file']
        data['ROLE'] = 'Test'
        data.drop(['Li', 'Robinson', 'Sun', 'Doddareddy', 'All Results', 'ID'], axis=1, inplace=True)
        data.shape

        # Remove internal duplicates
        print("From SDF test data size = " + str(data.shape[0]))
        data.drop_duplicates(subset='INCHI', keep='first', inplace=True)
        print("   -After removing dupplicates size = " + str(data.shape[0]))

        # Remove overlap with training data
        data = data[~data.INCHI.isin(training_data.INCHI)].dropna()
        data.to_pickle(pickleFile)
        print("   -After removing training overlap size = " + str(data.shape[0]))
    return data


"""
    Loads the Cronos transfer data
---
      test_data_file : the file to load
      training_data : the training data (used to remove duplicates)
      force : if true forces to reload from SD file

"""

def load_transfer_data(params, force=True):
    file = params['transfer_data_file']
    pickleFile = os.path.join("data", file + ".pkl")

    # use pre-converted pickle file if force = false
    # else reconvert the training data from the SDF file
    if not force and path.exists(pickleFile):
        data = pd.read_pickle(pickleFile)
        print("From pickle transfer data size = " + str(data.shape[0]))
    else:
        sdFile = os.path.join("data", file + ".sdf")
        data = PandasTools.LoadSDF(sdFile, molColName='MOLECULE')
        data['INCHI'] = [Chem.inchi.InchiToInchiKey(Chem.inchi.MolToInchi(mol)) for mol in data.MOLECULE]
        data['FP'] = data['MOLECULE'].apply(compute_fp_array)
        data['SOURCE'] = file
        data['ROLE'] = 'Transfer'
        data.to_pickle(pickleFile)
        print("From SDF transfer data size = " + str(data.shape[0]))
    return data


"""
    Samples data for the actual experiment
---
      training_full : the full training data
      test_full     : the full test data
      transfer_full : the full transfer data
             params : the experiment configuration
"""

def sample_data(training_full, test_full, transfer_full, params):
    # Split validation data from training data (same space)
    validationData = training_full.sample(frac=params['validation_ratio'], random_state=params['random_state'])
    print('Validation data size = ' + str(validationData.shape[0]))

    ###
    # Sample training data (if necessary)
    trainingData = training_full[~training_full.INCHI.isin(validationData.INCHI)].dropna()
    if params['training_size'] > 0 : trainingData = trainingData.sample(params['training_size'], random_state=params['random_state'])
    print('Training data size = ' + str(trainingData.shape[0]))

    ###
    # Sample test data (if necessary)
    if params['test_size'] > 0:
        testData = test_full.sample(params['test_size'], random_state=params['random_state'])
    else:
        testData = test_full

    print('Test data size = ' + str(testData.shape[0]))

    ###
    # Sample transfer data (if necessary)
    if params['transfer_size'] > 0:
        transferData = transfer_full.sample(params['transfer_size'], random_state=params['random_state'])
    else:
        transferData = transfer_full
    print('Transfer data size = ' + str(transferData.shape[0]))
    return trainingData, testData, transferData, validationData


"""
    Displays a data distribution
---
      data : the source data
      names: the labels of the categories 
      title: the title of the plot
"""

def display_distribution(data, names, title):
    fig = px.pie(data, names=names, hole=.2,
                 color_discrete_sequence=px.colors.qualitative.G10,
                title=title)
    fig.update_layout(width=500, height=500)
    
    fig = px.histogram(data, x='ACTIVITY', width=600,
                    height=400,title=title)
    showPlot(fig)

"""
    Clusters a data space into k subspaces
---
      data_space : the data space to cluster
            names: the labels of the categories 
           prefix: the cluster tag prefix
                k: the number of clusters
    smooth_factor: the cluster smoothing factor (mixing level)
           params: the experiment configuration
---
    cl;uster_data: the list of clusters
"""

def cluster_data_space(data_space, name, prefix, k, smooth_factor, params):
    X = np.asarray([fp for fp in data_space.FP])
    print('Clustering data k = ' + str(k))
    kmeans = KMeans(n_clusters=k, random_state=params['random_state']).fit(X)

    # Store the cluster indices and ID
    data_space['CLUSTER'] = kmeans.labels_ + 1
    data_space['CLUSTER_ID'] = prefix + data_space['CLUSTER'].astype(str)

    # Smooth the clusters by randomly changind the
    # cluster assignement of a fraction of the training data
    size = len(data_space.index)
    permutationCount = int(smooth_factor * size)
    random.seed(params['random_state'])

    for p in range(permutationCount):
        x = random.randrange(size)
        y = random.randrange(k)
        data_space['CLUSTER'][x] = y + 1
        data_space['CLUSTER_ID'][x] = prefix + str(y + 1)

    cluster_table = pd.DataFrame(columns=[name, 'Active', 'Inactive', 'Total', 'Proportion (%)'])
    size = len(data_space.index)

    # Teacher 0 = whole dataset
    cluster_data = []
    cluster_data.append(data_space)
    cluster_table = add_cluster_data(cluster_table, name, prefix, data_space, 0, size)

    # Add all the other teachers (one per cluster)
    for t in range(1, k + 1):
        data = data_space[data_space['CLUSTER'] == t]
        cluster_data.append(data)
        cluster_table = add_cluster_data(cluster_table, name, prefix, data, t, size)

    display(cluster_table)

    # Plot teacher distribution
    fig = go.Figure(
        data=[go.Bar(name=category, x=data_space.CLUSTER_ID.unique(),
                     y=[len(data_space.groupby(['CLUSTER', 'ACTIVITY']).groups[(cluster, category)]) for cluster in
                        data_space.CLUSTER.unique()])
              for category in data_space['ACTIVITY'].unique()])
    fig.update_layout(barmode='stack', width=800, height=600, title="Teacher distribution", title_x=0.5,
                      font=params['figure_font'])
    showPlot(fig)

    return cluster_data


"""
    Plot a t-SNE projection of a given data space
---
      datasets : the datasets forming the data spaces
            tag: the labels for the different datasets
          title: the title of the plot
         params: the experiment configuration
"""

def project_tsne(datasets, tag, title, params):
    data = pd.DataFrame()
    for dataset in datasets:
        data = pd.concat([data, dataset.sample(min(100, len(dataset)))])

    model = TSNE(n_components=2, random_state=params['random_state'], n_iter=params['tsne_iterations'])
    X = np.asarray([fp for fp in data.FP])
    x = model.fit_transform(X)

    data['TSNE1'] = x[:, 0]
    data['TSNE2'] = x[:, 1]

    x = pd.DataFrame(x)
    x.columns = ['TSNE1', 'TSNE2']

    fig = px.scatter(data, x="TSNE1", y="TSNE2", color=tag, color_discrete_sequence=px.colors.qualitative.G10,
                     opacity=0.75)

    fig.update_layout(width=700, height=600, title_text=title, font=params['figure_font'], title_x=0.5)
    fig.update_traces(marker=dict(size=10))
    showPlot(fig)


"""
    Plot a t-SNE projection of a given teacher cluster space
---
 teacher_space : the list of teacher data
         params: the experiment configuration
"""

def project_teacher_cluster_space(teacher_space, params):
    data = teacher_space.sample(params['tsne_size'], random_state=params['random_state'])
    model = TSNE(n_components=2, random_state=params['random_state'], n_iter=params['tsne_iterations'])
    X = np.asarray([fp for fp in data.FP])
    x = model.fit_transform(X)

    data['TSNE1'] = x[:, 0]
    data['TSNE2'] = x[:, 1]
    data['TAG'] = data['CLUSTER_ID']
    
    x = pd.DataFrame(x)
    x.columns = ['TSNE1', 'TSNE2']

    fig = px.scatter(data, x="TSNE1", y="TSNE2", color='TAG', color_discrete_sequence=px.colors.qualitative.G10,
                     opacity=0.75)

    font = dict(family="Arial",size=14,color="black")
    
    fig.update_layout(width=700, height=600, title_text="Teacher space", font=font, title_x=0.5,
                     paper_bgcolor='white', plot_bgcolor='white', legend_title_text='Teachers')
    fig.update_traces(marker=dict(size=10))    
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    
    showPlot(fig)

"""
    Plot a t-SNE projection of a given teacher activity pace
---
 teacher_space : the list of teacher data
         params: the experiment configuration
"""

def project_teacher_activity_space(teacher_space, params):
    data = teacher_space.sample(params['tsne_size'], random_state=params['random_state'])
    model = TSNE(n_components=2, random_state=params['random_state'], n_iter=params['tsne_iterations'])
    X = np.asarray([fp for fp in data.FP])
    x = model.fit_transform(X)

    data['TSNE1'] = x[:, 0]
    data['TSNE2'] = x[:, 1]

    x = pd.DataFrame(x)
    x.columns = ['TSNE1', 'TSNE2']

    #colors=['deepskyblue','#3365b5']
    colors=['deepskyblue','red']
    fig = px.scatter(data, x="TSNE1", y="TSNE2", color='ACTIVITY', color_discrete_sequence=colors,
                    opacity=0.75)
    

    font = dict(family="Arial",size=14,color="black")
    fig.update_layout(width=700, height=600, title_text="Teachers activity", font=font, title_x=0.5,
                     paper_bgcolor='white', plot_bgcolor='white', legend_title_text='Activity')
    fig.update_traces(marker=dict(size=10))    
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_traces(marker=dict(size=10))
    showPlot(fig)

"""
    Plot a t-SNE projection of a given set of data 
---
        datasets : the training data
           params: the experiment configuration
"""

def plot_data_space(datasets, params):
    
    for d in datasets:
        d[0]['TAG'] = d[1]

    reduced_sets = [d[0].sample(min(len(d[0]),d[2])) for d in datasets]
    space = pd.concat(reduced_sets)
    data = space.sample(n=params['tsne_size'], random_state=params['random_state'])
    model = TSNE(n_components=2, random_state=42, n_iter=params['tsne_iterations'])
    X = np.asarray([fp for fp in data.FP])
    x = model.fit_transform(X)
    data['TSNE1'] = x[:, 0]
    data['TSNE2'] = x[:, 1]

    x = pd.DataFrame(x)
    x.columns = ['TSNE1', 'TSNE2']

    fig2 = px.scatter(data, x="TSNE1", y="TSNE2", color='TAG', color_discrete_sequence=px.colors.qualitative.G10,
                      opacity=0.75)
    fig2.update_layout(width=700, height=600, title_text="t-SNE 2D Global space projection", font=params['figure_font'],
                       title_x=0.5)
    fig2.update_traces(marker=dict(size=10))
    showPlot(fig2)

"""
    Plot a t-SNE projection of a given transfer space
---
   training_data : the training data
       test_data : the test data
   transfer_data : the transfer data
           params: the experiment configuration
"""

def plot_transfer_space(training_data, test_data, transfer_data, params):
    #training_data['ROLE'] = training_data['CLUSTER_ID']
    spaceSize = min(len(transfer_data.index), 50000)
    space = pd.concat([training_data, test_data, transfer_data.sample(spaceSize, random_state=params['random_state'])])
    data = space.sample(n=params['tsne_size'], random_state=params['random_state'])
    model = TSNE(n_components=2, random_state=42, n_iter=params['tsne_iterations'])
    X = np.asarray([fp for fp in data.FP])
    x = model.fit_transform(X)

    data['TSNE1'] = x[:, 0]
    data['TSNE2'] = x[:, 1]

    x = pd.DataFrame(x)
    x.columns = ['TSNE1', 'TSNE2']

    colors=['DarkOrchid','DodgerBlue','DeepPink']
    fig = px.scatter(data, x="TSNE1", y="TSNE2", color='ROLE', color_discrete_sequence= colors,
                      opacity=0.75)
        
    font = dict(family="Arial",size=14,color="black")
    fig.update_layout(width=700, height=600, title_text="Data space", font=font, title_x=0.5,
                     paper_bgcolor='white', plot_bgcolor='white', legend_title_text='Activity')   
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_traces(marker=dict(size=10))
    showPlot(fig)

"""
    Build the teacher models given teacher data
---
   teacher_data : the list of teacher datasets
           params: the experiment configuration
"""

def build_teacher_models(teacher_data, params):
    teacher_models = []

    for t in tqdm(range(0, params['k'] + 1)):
        classifier = create_trained_classifier(params['teacher_algorithm'], teacher_data[t])
        teacher_models.append((classifier, len(teacher_data[t]), teacher_data[t]))
    return teacher_models


"""
    Crossvalidate the teachers
---
   teacher_data : the list of teacher datasets
           params: the experiment configuration
"""
def cross_validate_teachers(teacher_data, params):
    if params['details'] > 2 : 
        print("Performing teacher cross-validation...")
        teacher_cv_table = create_validation_table()
        algorithm = params['teacher_algorithm']

        for t in tqdm(range(0, params['k'] + 1)):
            classifier = create_classifier(algorithm)
            teacher_cv_table = add_classifier_cross_validation(teacher_cv_table, classifier, teacher_data[t], 'T' + str(t))

        # Plot the results
        fig = px.bar(teacher_cv_table, x='Model', y='MCC', color='MCC', width=600,
                    height=400, color_continuous_scale=params['figure_color_scale'], range_color=([0, 1]))
        fig.update_layout(title="Teachers internal 5x cross-validation (algo: " + algorithm + " mode:" + params['student_mode'] + ")", title_x=0.5,
                        font=params['figure_font'])
        fig.update_yaxes(range=[0, 1])
        showPlot(fig)

        display(teacher_cv_table)
    else : print("Skipped use details > 2 to run this section ")


"""
    Build the teacher models given teacher models
---
   teacher_models: the list of teacher models
        test_data: the test data 
           params: the experiment configuration
"""
def validate_teachers(teacher_models, test_data, params):
    print("Performing teacher external validation...")
    teacher_ev_table = create_validation_table()

    # For each teacher add a validation row
    for t in tqdm(range(0, params['k'] + 1)):
        classifier = teacher_models[t][0]
        teacher_ev_table = add_classifier_validation(teacher_ev_table, classifier, test_data, 'T-' + str(t), teacher_models[t][1])

    display(teacher_ev_table)

    # Plot the teacher validation performances
    fig = px.bar(teacher_ev_table, x='Model', y='MCC', color='MCC', width=600, height=400,
                 color_continuous_scale=params['figure_color_scale'], range_color=[0, 1])
    fig.update_layout(title="Teachers validation (algo: " + params['teacher_algorithm'] + " mode:" + params['student_mode'] + ")",
                      title_x=0.5, font=params['figure_font'])
    fig.update_yaxes(range=[0, 1])
    showPlot(fig, pause=True)

    # Compute the mean teacher performance
    # skip first teacher (T0 = whole training set)
    average_teacher_table = create_validation_table()
    average_teacher_table = average_teacher_table.append(
        pd.Series(['T-mean'], index=['Model']).append(teacher_ev_table.iloc[1:].mean(axis=0)), ignore_index=True)
    average_teacher_table = average_teacher_table.append(
        pd.Series(['T-max'], index=['Model']).append(teacher_ev_table.iloc[1:].max(axis=0).iloc[1:]), ignore_index=True)
    average_teacher_table = average_teacher_table.append(
        pd.Series(['T-min'], index=['Model']).append(teacher_ev_table.iloc[1:].min(axis=0).iloc[1:]), ignore_index=True)
    average_teacher_table = average_teacher_table.append(
        pd.Series(['T-std'], index=['Model']).append(teacher_ev_table.iloc[1:].std(axis=0).iloc[1:]), ignore_index=True)

    display(average_teacher_table)
    return teacher_ev_table, average_teacher_table



"""
    Ccreate the a new federated data initial table
---
     data : the Cronos dataset containinghte molecules and their fingerprint
"""

def create_federated_data(data):
    # Create new label table
    federated_data = pd.DataFrame()
    # Copy 'MOLECULE' and 'FP' columns
    federated_data['MOLECULE'] = [m for m in data['MOLECULE']]
    federated_data['FP'] = [fp for fp in data['FP']]
    return federated_data


"""
    Annotate the transfer data with the teacher models
---
            data : the Cronos dataset containinghte molecules and their fingerprint
"""
def annotate_transfer_data(transfer_data, teacher_models, teacher_data, params, force=False):
    pickleFile = os.path.join("data", params['cronos_label_file'] + ".pkl")
    k = params['k']

    # If we do not force to re-annotate the data then use the savec pickle file
    # otherwise annotate the transfer date using the teacher models
    if not force and path.exists(pickleFile):
        federated_data = pd.read_pickle(pickleFile)
        print("From pickle Cronos label table shape = " + str(federated_data.shape))
    else:
        print("Labelling the Cronos data...")

        # here we annotate directly the transfer_data (no copy)
        federated_data = transfer_data
        x = np.asarray([fp for fp in transfer_data.FP])

        # for each teacher 't' annotate the transfer data
        for t in tqdm(range(0, k + 1)):
            # compute prediction of teacher t
            classifier = teacher_models[t][0]
            y_pred = classifier.predict_proba(x)
            federated_data["C-T" + str(t)] = [("Active" if p[1] > p[0] else "Inactive") for p in y_pred]
            federated_data["P-T" + str(t)] = [(p[1] if p[1] > p[0] else p[0]) for p in y_pred]
            federated_data["Y-T" + str(t)] = [(p[0], p[1]) for p in y_pred]

            # compute weights of teacher t based on 8 closest neightbours
            data = teacher_data[t].sample(frac=0.2, random_state=params['random_state'])
            teacher_data_x = np.asarray([fp for fp in data.FP])
            knn = NearestNeighbors(n_neighbors=8)
            knn.fit(teacher_data_x)
            distances, indices = knn.kneighbors(x)
            federated_data["W-T" + str(t)] = [distances[index].mean() for index in range(len(transfer_data.index))]

        # Normalise the weights (absolute euclidian distances) to correspond to
        # similarities between 0 and 1
        print("Analysing teacher contribution similarities...")
        minWeight = min([federated_data["W-T" + str(t)].min() for t in range(1, k + 1)])
        if params['details'] > 3 : print(" Min:" + str(minWeight))
        maxWeight = max([federated_data["W-T" + str(t)].max() for t in range(1, k + 1)])
        if params['details'] > 3 : print(" Max:" + str(maxWeight))
        deltaWeight = maxWeight - minWeight

        # skip/ignore teacher 0 (full ChEMBL)
        federated_data["W-T" + str(0)] = 0

        # Scale the weights between (0,1)
        # and convert to similarity instead of distance s = 1-d
        print("Computing teacher contribution weights...")

        for t in tqdm(range(1, k + 1)):
            federated_data["W-T" + str(t)] = [pow(1 - (federated_data["W-T" + str(t)][index] - minWeight) / deltaWeight, 2)
                                           for index in transfer_data.index]

        print("Normalizing teacher contribution weights...")

        # Normalise the weights to sum up to 1 across the teachers
        #for index in tqdm(transfer_data.index):
        #    sumWeights = sum([federated_data["W-T" + str(t)][index] for t in range(1, k + 1)])

        #    for t in range(1, k + 1):
        #        federated_data["W-T" + str(t)][index] = federated_data["W-T" + str(t)][index] / sumWeights

        federated_data['sumWeights'] = [sum([federated_data["W-T" + str(t)][index] for t in range(1, k + 1)])  for index in transfer_data.index]

        for t in tqdm(range(1, k + 1)):
            federated_data["W-T" + str(t)] = federated_data["W-T" + str(t)] / federated_data['sumWeights']

        if params['details'] > 3 : print("This value : " + str(sum([federated_data["W-T" + str(t)][transfer_data.index[0]] for t in range(1, k + 1)]))
              + " should be very close to 1.0")

        #display(federated_data)

        # save to pickle format for future reuse
        federated_data.to_pickle(pickleFile)

        print("Cronos label table shape : " + str(federated_data.shape))

    return federated_data


"""
    Federate the annotated data 
---
    federated_data: the data to federate
          teachers: the list of teachers to federate against
---
           classes: the list of classes for each transfer instance
     probabilities: the list of probabilities for each instance
"""
def federate_labels(federated_data, teachers):
    # For each teacher, add its contribution
    #for index in federated_data.index:
    federated = [
            [sum([federated_data['Y-T' + str(t)][index][0] * federated_data['W-T' + str(t)][index] for t in teachers]/sum([federated_data['W-T' + str(t)][index] for t in teachers])),
             sum([federated_data['Y-T' + str(t)][index][1] * federated_data['W-T' + str(t)][index] for t in teachers]/sum([federated_data['W-T' + str(t)][index] for t in teachers]))]
        for index in federated_data.index]

    classes = [("Active" if p[1] > p[0] else "Inactive") for p in federated]
    probabilities = [(p[1] if p[1] > p[0] else p[0]) for p in federated]

    #display(federated)
    return classes, probabilities


###
# Compute the comtribution of each teacher in the federation
def compute_contributions(federated_data,  teachers):
    contributions = pd.DataFrame()
    contributions['Teacher'] = ['Teacher ' + str(t) for t in teachers]
    contributions['Contribution'] = [sum([federated_data['W-T' + str(t)][index] for index in federated_data.index]) for t in
                                     teachers]

    return contributions

"""
    Federate the annotated data for all teachers
---
    federated_data: the data to federate
            params: the experiment configuration
---
    federated_data: the completed federated data table
"""
def federate_teacher_annotations(federated_data, params):
    k = params['k']
    teachers = [t for t in range(1, k + 1)]

    # Federate using a similarity weighted average scheme
    classes, probabilites = federate_labels(federated_data, teachers)

    federated_data["C-F" + str(k)] = classes
    federated_data["P-F" + str(k)] = probabilites

    # Compute and plot the comtribution of each teacher in the federation
    contributions = compute_contributions(federated_data, teachers)
    fig = px.pie(contributions, values='Contribution', names='Teacher', hole=.2,
                 color_discrete_sequence=px.colors.qualitative.G10,
                 title='Contributions for federation mode :' + params['federated_student'])
    showPlot(fig)
    add_class_to_student_data(federated_data,params['federated_student'])
    display(federated_data.shape)
    return federated_data


"""
    Plots the annotation class distribution
---
    federated_data: the data to federate
            params: the experiment configuration
"""
def plot_annotation_distributions(data, width, height, params):
    students  = ['T' + str(t) for t in range(0, params['k'] + 1)]
    names  = ['F' + str(t) for t in range(0, params['k'] + 1)]
    students.append(params['federated_student'])
    categories = ['Active', 'Inactive']
    fig = go.Figure(
        data=[go.Bar(name=category, 
        x=names,
        y=[len(data.groupby(['C-' + student]).groups[(category)]) for student in students])
              for category in categories])
    
                      
    # Reduce opacity to see both histograms
    fig.update_layout(barmode='stack', width=width, height=height,
                      title="Activity distribution for individual and federated students",
                      title_x=0.5, font=params['figure_font'])
    fig.update_traces(opacity=0.5)
    showPlot(fig)


"""
    Computes the teacchrs class probability distibutions
---
    federated_data: the data to federate
            params: the experiment configuration
"""
def compute_teacher_probability_distributions(federated_data, params):
    k = params['k']
    confidences = pd.DataFrame()
    confidences['Teacher'] = ['T' + str(t) for t in range(0, k + 1)]
    confidences['min'] = [federated_data['P-T' + str(t)].min() for t in range(0, k + 1)]
    confidences['mean'] = [federated_data['P-T' + str(t)].mean() for t in range(0, k + 1)]
    confidences['max'] = [federated_data['P-T' + str(t)].max() for t in range(0, k + 1)]
    confidences['std'] = [federated_data['P-T' + str(t)].std() for t in range(0, k + 1)]

    #confidences = confidences.style.background_gradient(cmap=params['green_map'])
    display(confidences)

"""
    Plots the student annotation class distribution
---
    federated_data: the data to federate
            params: the experiment configuration
"""
def compute_student_probability_distribution(federated_data, params):
    confidences = pd.DataFrame()
    students = ['F' + str(params['k'])]
    confidences['Student'] = students
    confidences['min'] = [federated_data['P-' + s].min() for s in students]
    confidences['mean'] = [federated_data['P-' + s].mean() for s in students]
    confidences['max'] = [federated_data['P-' + s].max() for s in students]
    confidences['std'] = [federated_data['P-' + s].std() for s in students]

    confidences = confidences.style.background_gradient(cmap=params['green_map'])
    display(confidences)

"""
    Plots the annotation confidence distribution
---
    federated_data: the data to federate
            params: the experiment configuration
"""
def plot_confidence_distributions(data, width, height,  params):
    if params['details'] > 1 : 
        students = ['T' + str(i) for i in range(0, params['k'] + 1)]
        students.append(params['federated_student'])

        fig = make_subplots(rows=len(students) // 4 + 1, cols=4, subplot_titles=students, horizontal_spacing=0.1)
        colIndex = 0
        rowIndex = 1

        for student in students:
            colIndex = colIndex + 1
            if colIndex > 4:
                colIndex = 1
                rowIndex = rowIndex + 1
            fig.add_trace(go.Histogram(marker=dict(color='blue'), name=student + '(Active)',
                                    x=data[data['C-' + student] == 'Active']['P-' + student]), row=rowIndex,
                        col=colIndex)
            fig.add_trace(go.Histogram(marker=dict(color='red'), name=student + '(Inactive)',
                                    x=data[data['C-' + student] == 'Inactive']['P-' + student]), row=rowIndex,
                        col=colIndex)

        # Reduce opacity to see both histograms
        fig.update_layout(barmode='overlay', width=width, height=height,
                        title="Decidability distribution for federated students",
                        title_x=0.5, font=params['figure_font'])
        fig.update_traces(opacity=0.5)
        showPlot(fig)
    else : print("Skipped use details > 1 to run this section ")

"""
    Creates a claissifier model given a selection of data
---
    selection: the selectedc training data
    algorithm: the learning algorithm
      student: the student annotation to use
---
   classifier: the resulting classifier
"""
def create_selection_model(selection, algorithm, student):
    X = np.asarray([fp for fp in selection.FP])
    Y = np.asarray([1 if c == "Active" else 0 for c in selection['C-' + student]])
    classifier = create_classifier(algorithm)
    classifier.fit(X, Y)
    return classifier
    

"""
    Selects a subset of student data
---
    federated_data: the federated_data to select from
      student: the student annotation to use
      size: the number of instances to selecct
      mode: the selection mode
      params: the experiment configuration
---
   student_selection: the selected student data
"""
def select_student_data(federated_data, student, size, mode, params):
    # Balanced
    if mode == 'balanced':
        table = federated_data.sample(frac=1, random_state=params['random_state']) # shuffling
        actives = table[table['C-' + student] == 'Active'].head(int(size / 2))
        inactives = table[table['C-' + student] == 'Inactive'].head(int(size / 2))

    # Random
    elif mode == 'random':
        actives = federated_data.sample(int(size / 2), random_state=params['random_state'])
        inactives = federated_data.sample(int(size / 2), random_state=params['random_state'])

    # Best
    elif mode == 'best':
        table = federated_data.sort_values('P-' + student, ascending=False) # sorting
        actives = table[table['C-' + student] == 'Active'].head(int(size / 2))
        inactives = table[table['C-' + student] == 'Inactive'].head(int(size / 2))

    # Worst
    elif mode == 'worst':
        table = federated_data.sort_values('P-' + student, ascending=True) # sorting (reverse)
        actives = table[table['C-' + student] == 'Active'].head(int(size / 2))
        inactives = table[table['C-' + student] == 'Inactive'].head(int(size / 2))

    else:
        raise ValueError('{foo} wrong mode, use "random" or "best" or "worst" or "mixed-positive" or "mixed-negative"'.format(foo=mode))

    if str(len(actives.index)) != str(len(inactives.index)):
        if params['details'] > 3 : print("/!\ Student data not balanced for selection mode '" + mode + "', " + str(
            len(actives.index)) + " actives and  " + str(len(inactives.index)) + " inactives")

    student_selection = pd.concat([actives, inactives])
    student_selection = student_selection[~student_selection.index.duplicated(keep='first')]
    add_class_to_student_data(student_selection, student)

    return student_selection

"""
    Creates a student model
---
 federated_data: the federated_data to select from
        student: the student annotation to use
           size: the number of instances to selecct
           mode: the selection mode
      algorithm: the learning algorithm
         params: the experiment configuration
---
   classifier: the student classifier
"""
def create_student_model(federated_data, student, size, mode, algorithm, params):
    selection = select_student_data(federated_data, student, size, mode, params)
    classifier = create_selection_model(selection, algorithm, student)
    return classifier


"""
    Select federated data for a given list of teachers
---
 federated_data: the federated_data to select from
       teachers: the list of teachers to use
           size: the number of instances to selecct
         params: the experiment configuration
---
   selection: the list of selected instances
"""
def select_federated_data(federated_data, teachers, size, params):
    selection = select_student_data(federated_data, params['federated_student'], size,  params['student_mode'], params)
    classes, probabilities = federate_labels(selection, teachers)
    selection['CLASS'] =  [(1 if c == "Active" else 0 ) for c in classes]
    return selection

"""
    Creates a federated model for a given list of teachers
---
 federated_data: the federated_data to select from
       teachers: the list of teachers to use
           size: the number of instances to selecct
      algorithm: the learning algorithm
         params: the experiment configuration
---
   classifier: the federated classifier
"""
def create_federated_model(federated_data, teachers, size, algorithm, params):
    data = select_federated_data(federated_data, teachers, size, params)
    model = create_trained_classifier(algorithm, data)
    return model


"""
    Creates a student validation table for a given list of students
---
       students: the list of target students
           size: the number of instances to selecct
           mode: the federation mode
      algorithm: the learning algorithm
      test_data: the test data
---
   classifier: the federated classifier
"""

def create_student_validation_table(students, size, mode, algorithm, test_data):
    studentEVTable = create_validation_table()

    for student in tqdm(students):
        classifier = create_student_model(student, size, mode, algorithm)
        studentEVTable = add_classifier_validation(studentEVTable, classifier, test_data, 'S-' + student, size)

    return studentEVTable


"""
    adds a single validation row to a vaidation table
---
          table: the validation table to append
           size: the number of instances to select
           mode: the federation mode
      algorithm: the learning algorithm
        student: the student to use
      test_data: the test data
---
   classifier: the federated classifier
"""

def add_single_student_validation_row(table, size, mode, algorithm, student, test_data):
    studentSelection = select_student_data(student, size, mode)
    X = np.asarray([fp for fp in studentSelection.FP])
    Y = np.asarray([1 if c == "Active" else 0 for c in studentSelection['C-' + student]])
    classifier = create_classifier(algorithm)
    classifier.fit(X, Y)
    table = add_classifier_validation(table, classifier, test_data, algorithm + '-S-' + str(student) + "-" + mode,
                                    len(studentSelection.index))
    return table


"""
    creates a benchmark table for a 
---
 federated_data: the validation table to append
           size: the number of instances to select
           mode: the federation mode
      test_data: the test data
         params: the experiment configuration
---
   student_validation_table: the validation table
"""

def create_benchmark_table(federated_data, size, mode, test_data, params):
    
    # Individual Teacher-Student relation + federated student
    students = ['T' + str(i) for i in range(0, params['k'] + 1)]
    students.append(params['federated_student'])
    student_validation_table = create_validation_table()

    # for each individual student 
    for student in tqdm(students):
        classifier = create_student_model(federated_data, student, size, mode, params['student_algorithm'], params)
        student_validation_table = add_classifier_validation(student_validation_table, classifier, test_data, 'S-' + student, size)

    return student_validation_table


"""
    Main Cronos benchmark 
---
 federated_data: the validation table to append
           size: the number of instances to select
           mode: the federation mode
      test_data: the test data
         params: the experiment configuration
---
   student_validation_table: the validation table
"""

def benchmark(federated_data, teacher_models, test_data, params):

    # params
    k = params['k']
    mode = params['student_mode']
    size = params['student_size']
    student = params['federated_student']
    algorithm = params['student_algorithm']
    prefix = "'T/S/H-'"
    # Plot the teacher, student and hybrid performance
    fig = go.Figure()

    # Teachers performance
    teacher_validation_table, average_teacher_table = validate_teachers(teacher_models, test_data, params)

    fig.add_trace(go.Bar(
        x=[prefix + str(i) for i in range(0, k + 1)] + [student],
        y=[teacher_validation_table.iloc[i]['MCC'] for i in range(0, k + 1)] + [average_teacher_table.iloc[0]['MCC']],
        name='Teacher',
        marker_color='dodgerblue'))

    # Student performance
    print("Benchmarking student for mode '" + mode + "'")

    student_validation_table = create_benchmark_table(federated_data, size, mode, test_data, params)
    student_validation_table = student_validation_table.append(
        pd.Series(['S-mean'], index=['Model']).append(student_validation_table.iloc[1:].mean(axis=0)), ignore_index=True)
    display(student_validation_table)
    
    fig.add_trace(go.Bar(
        x=[prefix + str(i) for i in range(0, k + 1)] + [student],
        y=[student_validation_table.iloc[i]['MCC'] for i in range(0, k + 2)],
        name='Student',
        marker_color='plum'))

    # Hybrid models (student + teacher)
    hybrid_models = []
    hybrid_data_list = []
    students = ['T' + str(i) for i in range(0, params['k'] + 1)]
    hybrid_validation_table = create_validation_table()

    student_data = select_student_data(federated_data, student, size, mode, params)
    student_data['CLASS'] = [1 if c == 'Active' else 0 for c in student_data['C-' +  student]]

    for i in tqdm(range(0, k + 1)):
        teacher_data = teacher_models[i][2]
        hybrid_data = pd.concat([student_data, teacher_data])
        hybrid_model = create_trained_classifier(algorithm, hybrid_data)
        hybrid_validation_table = add_classifier_validation(hybrid_validation_table, hybrid_model, test_data,'H' + str(i), size)
        hybrid_data_list.append(hybrid_data)
        hybrid_models.append(hybrid_model)

    teacher_data = teacher_models[0][2]
    hybrid_data = pd.concat([student_data, teacher_data])
    hybrid_model = create_trained_classifier(algorithm, hybrid_data)
    hybrid_validation_table = add_classifier_validation(hybrid_validation_table, hybrid_model, test_data,'H0-' + student, size)
    display(hybrid_validation_table)

    fig.add_trace(go.Bar(
        x=[prefix + str(i) for i in range(0, k + 1)] + [params['federated_student']],
        y=[hybrid_validation_table.iloc[i]['MCC'] for i in range(0, k + 2)],
        name='Hybrid',
        marker_color='mediumvioletred'))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=-45,
                      title_text="Benchmark Student/Teacher algo:" + params['student_algorithm'] + " mode:" + params['student_mode'] + " (" + str(
                          params['student_size']) + " data)",
                      title_x=0.5, font=params['figure_font'])
    fig.update_yaxes(range=[0, 0.75])
    showPlot(fig)

    return student_validation_table, teacher_validation_table, hybrid_data_list, hybrid_models


"""
    Add class information to federated data table for a given student
---
           data: the validation table to append
        student: the student to focus on
---
           data: the ammended federated data
"""

def add_class_to_student_data(data, student='F8'):
    data['CLASS'] = [1 if activity == 'Active' else 0 for activity in data['C-' + student]]
    data['ACTIVITY'] = data['C-' + student]
    return data


"""
    Prints the experiment configuration
---
           params: the experiment configuration
             keys: the settings to print
"""
def print_params(params, keys):
    for k in keys :
        print(k + " = " + str(params[k]))

"""
    Plots the cqategory distribution for a list of datasets
---
       datasets: the datasets to profile
        params: the experiment configuration
"""

def plot_distribution_set(datasets, params):
    data = pd.concat(datasets)

    ###
    # Plot teacher distribution
    fig = go.Figure(
        data=[go.Bar(name=category, x=data.TAG.unique(),
                     y=[len(data.groupby(['TAG', 'ACTIVITY']).groups[(tag, category)]) for tag in
                        data.TAG.unique()])
              for category in data['ACTIVITY'].unique()])
    fig.update_layout(barmode='stack', width=450, height=400, title="Data distribution", title_x=0.5,
                      font=params['figure_font'])
    showPlot(fig)



"""
    Validates across domains
---
    teacher_models: the teacher models
    federated_data: the federated data
         test_data: the test data
            params: the experiment configuration
---
        domain_average_table: the average performance across domain
        domain_table: the detailes domain performance table
"""
def cross_domain_validate(teacher_models, federated_data, test_data, params):
    if params['details'] > 2 :
        # tag federated data
        federated_data['TAG'] = 'federated'

        # init output variables
        domain_table = create_validation_table()
        loto_table = create_validation_table()

        # retrieve parameters
        k = params['k']
        student_size = params['student_size']
        student_mode = params['student_mode']
        student_algorithm = params['student_algorithm']
        federated_student = params['federated_student']

        # build the set of all teachers
        all_teachers = [t for t in range(1, k + 1)]

        # start with the full federated student
        student_data = select_student_data(federated_data, federated_student, student_size, student_mode, params)
        student_model = create_trained_classifier(student_algorithm, student_data)

        # validate the full student
        loto_table = add_classifier_validation(loto_table, student_model, test_data, federated_student, student_size)

        print("'Running Leave One Teacher Out' and 'Domain Adapatation' experiments")
        
        # use each teacher data as a target domain in turn
        for teacher in tqdm(all_teachers):
            single_table = create_validation_table()

            # compute the list of the other teachers
            other_teachers = [ i for i in all_teachers if i != teacher]

            # the validation domain is the domain of the selected teacher
            # we split this domain into target data (propsective data)
            target_domain = teacher_models[teacher][2]
            target_domain['TAG'] = 'Target-Domain' + str(teacher)

            # global data is the rest of the data (other teachers)
            global_data = pd.concat([teacher_models[i][2] for i in other_teachers])
            global_data['TAG'] = 'Global-Domain' + str(teacher)

            # compute the student model by federating all the other teachers
            student_data = select_federated_data(federated_data, other_teachers, student_size, params)
            student_data['TAG'] = 'Student-Domain' + str(teacher) 
            student_model = create_trained_classifier(student_algorithm, student_data)

            # validate the student against the benchmark data (Leave One Teacher Out)
            data = student_data
            model = student_model
            loto_table = add_classifier_validation(loto_table, model, test_data, 'No-T' + str(teacher), len(data))

            # validate the student against the target data (Cross Domain Validation)
            row = create_validation_row(domain_table, model, target_domain, 'S' , len(data))
            single_table = single_table.append(row, ignore_index=True)
            domain_table = domain_table.append(row, ignore_index=True)

            # space / distribution
            #project_tsne([cap(target_data), cap(global_data), cap(student_data), cap(genetic_data)],'TAG', 't-SNE projection-T' + str(teacher) , params)
            #plot_distribution_set([cap(target_domain), cap(global_data), cap(student_data), cap(genetic_data)], params)

            # validate each other teacher against the target domain 
            for other_teacher in all_teachers :
                if other_teacher != teacher:

                    teacher_table = create_validation_table()
                    other_teacher_data = teacher_models[other_teacher][2]
                    other_teacher_data['TAG'] = 'T' + str(teacher)

                    # teacher
                    data = pd.concat([other_teacher_data])
                    model = create_trained_classifier(student_algorithm, data)
                    row = create_validation_row(domain_table, model, target_domain, 'T', len(data))
                    single_table = single_table.append(row, ignore_index=True)
                    domain_table = domain_table.append(row, ignore_index=True)
                    teacher_table = teacher_table.append(row, ignore_index=True)

                    # hybrid (teacher + student)
                    data = pd.concat([other_teacher_data, student_data])
                    model = create_trained_classifier(student_algorithm, data)
                    row = create_validation_row(domain_table, model, target_domain, 'H' , len(data))
                    single_table = single_table.append(row, ignore_index=True)
                    domain_table = domain_table.append(row, ignore_index=True)
                    teacher_table = teacher_table.append(row, ignore_index=True)

            # Display single domain intermetiate performances
            #single_average_table = create_validation_table()

            #for model in single_table['Model'].unique():
            #    single_average_table = add_replicate_table_row(single_average_table, single_table, model)

            #single_average_table = single_average_table.sort_values('MCC', ascending=False)
            #display(single_average_table)

            # Plot the performances for this domain
            #fig = px.bar(single_average_table, x='Model', y='MCC', error_y ='MCC-C95', color='MCC', width=1200, height=600,
            #            color_continuous_scale=params['figure_color_scale'], range_color=[0, 1])
            #fig.update_layout(title="Domain validation for domain D" + str(teacher) + " (algo: " + " mode:" + params['student_mode'] + student_algorithm + ")",
            #                title_x=0.5, font=params['figure_font'])
            #fig.update_yaxes(range=[0, 1])
            #showPlot(fig)

        display(loto_table)
        
        # Plot the student external validation performances
        fig = px.bar(loto_table, x='Model', y='MCC', color='MCC', width=1200, height=600,
                    color_continuous_scale=params['figure_color_scale'], range_color=[0, 1])
        fig.update_layout(title="Leave one teacher out validation (algo: " + student_algorithm + " mode:" + params['student_mode'] + ")",
                        title_x=0.5, font=params['figure_font'])
        fig.update_yaxes(range=[0, 1])
        showPlot(fig, pause=True)

        domain_table = domain_table.sort_values('MCC', ascending=False)
        domain_average_table = create_validation_table()

        for model in domain_table['Model'].unique():
            domain_average_table = add_replicate_table_row(domain_average_table, domain_table, model)

        domain_average_table = domain_average_table.sort_values('MCC', ascending=False)
        display(domain_average_table)

        # Plot the student validation performances
        fig = px.bar(domain_average_table, x='Model', y='MCC', error_y='MCC-C95', color='MCC', width=1200, height=600,
                    color_continuous_scale=params['figure_color_scale'], range_color=[0, 1])
        fig.update_layout(title="Domain Adaptation summary (algo: " + student_algorithm + " mode:" + params['student_mode'] + ")",
                        title_x=0.5, font=params['figure_font'])
        showPlot(fig)
    else : print("Skipped use details > 2 to run this section ")

"""
    Plots the impact of the number of teachers
---
    label_data: the federated data
     test_Data: the test data
        params: the experiment configuration

"""
def benchmark_teacher_count(label_table, teacher_validation_table, test_data, params):
    if params['details'] > 1 :
        print("Computing incremental teacher federation...")

        # params
        algorithm = params['student_algorithm']
        size = params['student_size']
        k = params['k']
        teachers = [t for t in range(1, k + 1)]
        
        ###
        # Direct order
        students_table = create_validation_table()

        # Find the list of teacher in their order of increasing performance
        teachers.sort(key = lambda t: teacher_validation_table['MCC'][t])

        # For each number of teacher
        for n_teachers in tqdm(range(1, k + 1)):
            replicate_table = create_validation_table()
            # replicate the validation

            for replicate in range(10*params['replicate_count']):
                teacher_selection = random.sample(teachers, k=n_teachers)
                #print("Replicate:" + str(teacher_selection))

                student_model = create_federated_model(label_table, teacher_selection, size, algorithm, params)
                replicate_table = add_classifier_validation(replicate_table, student_model, test_data, 'F-' + str(n_teachers), size)

            #display(replicate_table)
            students_table = add_replicate_table_row(students_table, replicate_table, 'F-' + str(n_teachers))

        display(students_table)

        fig = px.bar(students_table, x='Model', y='MCC', error_y='MCC-C95', color='MCC', width=600, height=400,
                    color_continuous_scale=params['figure_color_scale'], range_color=[0, 1])
        fig.update_layout(
            title="Incremental Federated Student (" + str(size) + " data) with algo: " + algorithm + " mode:" + params['student_mode'] ,
            title_x=0.5, font=params['figure_font'])
        fig.update_yaxes(range=[0, 1])
        showPlot(fig)
    else : print("Skipped use details > 1 to run this section ")

"""
    Plots the impact of  size of the student
---
    label_data: the federated data
     test_Data: the test data
         sizes: the size the consider
        params: the experiment configuration

"""
def benchmark_student_size(label_table, training_data, test_data, params): 
    sizes = params['student_sizes']
    student = params['federated_student']
    mode = params['student_mode']
    algorithm=params['student_algorithm']
    return benchmark_student_size_custom(label_table, training_data, test_data, sizes , student , mode, algorithm, params)                                         
    
def benchmark_student_size_custom(label_table, training_data, test_data, sizes, student, mode, algorithm, params): 
    if params['details'] > 1 :
        
        student_size_table = create_validation_table()
        student_size_table['TAG'] = 'None'

        ###
        # Training data

        print("Scanning full training size impact ...")

        # for each size
        for size in tqdm(sizes):
            if(size < len(label_table)):
                replicate_table = create_validation_table()

                for replicate in range(10*params['replicate_count']):
                    
                    table = label_table.sample(frac=1) # free shuffling (no seed)
                    actives = table[table['C-' + student] == 'Active'].head(int(size / 2))
                    inactives = table[table['C-' + student] == 'Inactive'].head(int(size / 2))
                    data = pd.concat([actives, inactives])
                    
                    classifier = create_trained_classifier(params['student_algorithm'], data)
                    #classifier = create_student_model(label_table,  student, size,  mode, algorithm, params)
                    replicate_table = add_classifier_validation(replicate_table, classifier, test_data, 'S-' + str(size), size)

                student_size_table = add_replicate_table_row(student_size_table, replicate_table, 'S-' + str(size))
                student_size_table['TAG'][len(student_size_table) - 1] = 'Student'

        ###
        # Student data

        print("Scanning student training size impact ...")

        # for each size
        for size in tqdm(sizes):
            if(size < len(training_data)):
                replicate_table = create_validation_table()
                
                for replicate in range(10*params['replicate_count']):
                                    
                    # sample random balanced
                    table = training_data.sample(frac=1) # free shuffling (no seed)
                    actives = table[table['ACTIVITY'] == 'Active'].head(int(size / 2))
                    inactives = table[table['ACTIVITY'] == 'Inactive'].head(int(size / 2))
                    data = pd.concat([actives, inactives])
                    
                    classifier = create_trained_classifier(params['student_algorithm'], data)
                    replicate_table = add_classifier_validation(replicate_table, classifier, test_data, 'T-' + str(size), size)

                student_size_table = add_replicate_table_row(student_size_table, replicate_table, 'T-' + str(size))
                student_size_table['TAG'][len(student_size_table) - 1] = 'Training'

        display(student_size_table)

        # plot as bar plot (with errors)
        fig = px.bar(student_size_table, x='Model', y='MCC', error_y='MCC-C95', color='TAG', width=600, height=400,
                    color_continuous_scale=params['figure_color_scale'], range_color=[0, 1])
        fig.update_layout(
            title="Student Size Impact (algo: " + params['student_algorithm'] + " mode:" + params['student_mode'] + ')',
            title_x=0.5, font=params['figure_font'])
        fig.update_yaxes(range=[0, 1])
        showPlot(fig, pause=True)

        # plot as line plot (without errors)
        fig = px.line(student_size_table, x='Size', y='MCC', line_shape='spline', color='TAG', width=600, height=400)
        fig.update_layout(
            title="Size Impact (algo: " + params['student_algorithm'] + " mode:" + params['student_mode'] + ')',
            title_x=0.5, font=params['figure_font'])
        fig.update_yaxes(range=[0, 1])
        showPlot(fig)
        
        return student_size_table
    else : print("Skipped use details > 1 to run this section ")
    
    
# AD Stuff below this line ==============================================================================================

def ADbenchmark(teacher_data, hybrid_data, teacher_models, hybrid_models, test_data, radius = 2):
    print("Gathering Feature Dictionaries")
    print("teacher")
    teacherFD = ADCalculateFD(teacher_data, radius)
    print("hybrid")
    hybridFD = ADCalculateFD(hybrid_data, radius)
    
    print("Appending AD information to table")
    ADAppendDomainColumn(test_data, teacherFD, "Domain_T", radius)
    ADAppendDomainColumn(test_data, hybridFD, "Domain_H", radius)
    
    print("Predicting teacher models")
    ADAppendPredictionColumn(test_data, teacher_models, "Predicted_T")
    print("Predicting hybrid models")
    ADappendHybridPredictionColumn(test_data, hybrid_models, "Predicted_H")
    
    for mdl in hybrid_models:
        print(mdl[2])
    
    print("count of compounds brought into domain")
    domain_validate_test = []
    for i in range(0, len(teacher_models)):
        teacherColName = "Domain_T"+ str(i)
        hybridColName = "Domain_H" + str(i)
        domain_validate_test.append(test_data[test_data[teacherColName] == False][test_data[hybridColName] == True])
        print("HM_" + str(i) + ": " + str(domain_validate_test[i].__len__()))
    
    print("Brought into domain in hybrid:")
    display(ADValidate(domain_validate_test))
    
    validation_data_multiplex = []

    for i in range(0,9):
        validation_data_multiplex.append(test_data)
    
    print("All predictions")
    display(ADValidate(validation_data_multiplex))
    
    teacher_val_in_data = []

    for i in range(0,len(teacher_models)):
        teacherColName = "Domain_T" + str(i)
        teacher_val_in_data.append(test_data[test_data[teacherColName] == True])

    #print(len(teacher_val_in_data))

    hybrid_val_in_data = []

    for i in range(0,len(hybrid_models)):
        hybridColName = "Domain_H" + str(i)
        hybrid_val_in_data.append(test_data[test_data[hybridColName] == True])
    
    #print(len(hybrid_val_in_data))
    
    print("Teacher AD In:")
    display(ADValidate(teacher_val_in_data))

    print("Hybrid AD In:")
    display(ADValidate(hybrid_val_in_data))
    

'''
Takes in a list of tables with predicted and experimental data (as a CLASS so 1 or 0) and does validation statistics and outputs the performance 
of the Teacher and Hybrid models against the data supplied

returns a summary table of the results each rowname is prepended with the teacher or hybrid row prefix as required
'''
def ADValidate(domain_test, teacherRowPrefix = "TM_", hybridRowPrefix = "HM_", teacherColPrefix = "Predicted_T", hybridColPrefix = "Predicted_H"):
    val_table = create_validation_table()

    for i in range(0, len(domain_test)):
        teacherRowName = teacherRowPrefix + str(i)
        hybridRowName = hybridRowPrefix + str(i)
        teacherPredName = teacherColPrefix + str(i)
        hybridPredName = hybridColPrefix + str(i)
        val_table = add_validation_row(val_table, domain_test[i].CLASS, domain_test[i][teacherPredName], teacherRowName, len(domain_test[i]))
        val_table = add_validation_row(val_table, domain_test[i].CLASS, domain_test[i][hybridPredName], hybridRowName, len(domain_test[i]))
        '''
        val_table = val_table.append(compute_validation_row(val_table, domain_test[i].CLASS, domain_test[i][teacherPredName], teacherRowName, len(domain_test[i])), ignore_index = True)
        val_table = val_table.append(compute_validation_row(val_table, domain_test[i].CLASS, domain_test[i][hybridPredName], hybridRowName, len(domain_test[i])), ignore_index = True)
        '''
    return val_table

    
'''
takes in the test_data and a list of FeatureDictionaries and appends domain columns onto the test_data dataframe

---
    test_data: data table to append the domain column
    featureDict: List of FeatureDictionaries of the chemical space to use as the AD
    colPrefix: prefix for the column names
    radius (opt): the radius to use for fingerprinting default = 2
---
    no return appends onto supplied table
'''
def ADAppendDomainColumn(test_data, featureDict, colPrefix, radius = 2):
    for i in range(0, len(featureDict)):
        colname = colPrefix + str(i)
        print(colname)
        test_data[colname] = test_data['MOLECULE'].apply(lambda mol: ADMolecule(mol, featureDict[i], radius))
    

'''
Takes in a dataframe and appends predicted columns for the supplied models from a model_list

---
    test_data: data table on which the predictions will be appended
    model_list: a list of models to use to predict, eahc one will be used in turn
    colPrefix: a String of the prefix to use for the appended column
---
    no return appends onto supplied data table

'''
def ADAppendPredictionColumn(test_data, model_list, colPrefix):
    for i in range(0, len(model_list)):
        X = np.asarray([fp for fp in test_data.FP])
        colname = colPrefix + str(i)
        print(colname)
        test_data[colname] = model_list[i][0].predict(X)
        

def ADappendHybridPredictionColumn(test_data, model_list, colPrefix):
    for i in range(0, len(model_list)):
        X = np.asarray([fp for fp in test_data.FP])
        colname = colPrefix + str(i)
        print(colname)
        test_data[colname] = model_list[i].predict(X)          
'''
Takes in the list of teachers 0 ... k and returns the FeatureDictionary for them also as a list 0 ... k

---
    teacherData: list of dataframes with a structure in the MOLECULE column
    radius (optional): a radius to use when fingerprinting
---
    teacherFP: a list of FeatureDictionaries containing all of the set bits from the dataframe
'''
def ADCalculateFD(datasets, radius = 2):
    teacherFD = []
    for i in range(0, len(datasets)):
         teacherFD.append(ADTrainFingerprint(datasets[i], radius))
    return teacherFD
    
    
""" 
Takes in the supplied dataset and the radius (optional default 2) and returns a FeatureDictionary that is the result of the 
sum of all of the fingerprints in the dataset
"""
def ADTrainFingerprint(dataset, radius = 2):
    column = dataset['MOLECULE'].array
    fp = _ADFingerprint(column[0], radius)
    for mol in column[1:]:
        fp += _ADFingerprint(mol, radius)
    return fp


"""
you supply the molecule of question and a FeatureDictionary (see ADTrainFingerprint) and an
optional radius (default 2) and it returns a boolean of if the molecule is inDomain or not inDomain means that all of the bits in
the molecule are present in the training space
"""
def ADMolecule(molecule, featureDict, radius = 2):
    queryFP = _ADFingerprint(molecule, radius)
    return ADinDomain(queryFP, featureDict)


"""  
performs an and operation between the query and the featureDict and makes sure that the cardinality between teh resulting matchFP is the same as the query FP
"""
def ADinDomain(queryFP, featureDict):
    match = queryFP & featureDict
    return _FPsameLength(match, queryFP)


'''
prints to the console the size of all of the Feature dictionaries in the supplied list

---
    FeatureDictionaryList: a list of featuredictionaries
    
no return
'''
def ADPrintFDSize(FeatureDictionaryList):
    for FD in FeatureDictionaryList:
        print(len(FD.GetNonzeroElements()))

        
'''
Takes in a table with domain columns (boolean) and selects only the data where the row is OOD for the teacher but ID for the hybrid
The domain columns must be names of the structure Prefix## where prefix is the prefix of the column name ie Domain_T and ## is an incremental integer for the count of how many columns are present.
 
 ---
     data: table of data with domain columns
     columnCountL number of columns to use for each teacher and hybrid selection ie T0 ... T10 would be 11
     teacherPrefix: prefix of the domain columns for the teacher 
     hybridPrefix: prefix of the domain columns for the hybrid
 ---
 
 returns a list of data_Table subselected as described
'''
def ADGetDomainSelectedData(data, columnCount = 9, teacherPrefix = "Domain_T", hybridPrefix = "Domain_H"):
    domain_test = []
    for i in range(0, columnCount):
        teacherColName = teacherPrefix + str(i)
        hybridColName = hybridPrefix + str(i)
        domain_test.append(data[data[teacherColName] == False][data[hybridColName] == True])
            
    return domain_test


'''
gets the fingerprint of a molecule
allow for easy changing to adjust the fingerprint for use in calculating the AD
'''
def _ADFingerprint(molecule, radius, features = True):
    return AllChem.GetMorganFingerprint(molecule, radius, useFeatures=features)


"""
checks that the cardinality of the two fingerprints is the same  ie same number of nonzero elements
"""
def _FPsameLength(fp1, fp2):
    return len(fp1.GetNonzeroElements()) == len(fp2.GetNonzeroElements())
