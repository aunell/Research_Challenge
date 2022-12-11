from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from data import load_adult, preprocess_adult
from metrics import DP, FTU
from train import train_decaf
# import tensorflow as tf

import tensorflow as tf

tf.config.run_functions_eagerly(True)
dataset = load_adult()
dataset = preprocess_adult(dataset)
dataset.head()
dataset_train, dataset_test = train_test_split(dataset, test_size=2000, stratify=dataset['income'])

# Define DAG for Adult dataset
dag = [
    ['income', 'education'],
    # ['income', 'relationship'], 
    
    # Edges from race
    ['race', 'occupation'],
    ['race', 'income'],
    ['race', 'hours-per-week'],
    ['race', 'education'],
    ['race', 'marital-status'],

    # Edges from age
    ['age', 'occupation'],
    ['age', 'hours-per-week'],
    ['age', 'income'],
    ['age', 'workclass'],
    ['age', 'marital-status'],
    ['age', 'education'],
    ['age', 'relationship'],
    
    # Edges from sex
    ['sex', 'occupation'],
    ['sex', 'marital-status'],
    ['sex', 'income'],
    ['sex', 'workclass'],
    ['sex', 'education'],
    ['sex', 'relationship'],
    
    # Edges from native country
    ['native-country', 'marital-status'],
    ['native-country', 'hours-per-week'],
    ['native-country', 'education'],
    ['native-country', 'workclass'],
    ['native-country', 'income'],
    ['native-country', 'relationship'],
    
    # Edges from marital status
    ['marital-status', 'occupation'],
    ['marital-status', 'hours-per-week'],
    ['marital-status', 'income'],
    ['marital-status', 'workclass'],
    ['marital-status', 'relationship'],
    ['marital-status', 'education'],
    
    # Edges from education
    ['education', 'occupation'],
    ['education', 'hours-per-week'],
    ['education', 'income'],
    ['education', 'workclass'],
    ['education', 'relationship'],
    
    # All remaining edges
    ['occupation', 'income'],
    ['hours-per-week', 'income'],
    ['workclass', 'income'],
    ['relationship', 'income']
]

def dag_to_idx(df, dag):
    """Convert columns in a DAG to the corresponding indices."""

    dag_idx = []
    for edge in dag:
        dag_idx.append([df.columns.get_loc(edge[0]), df.columns.get_loc(edge[1])])

    return dag_idx

# Convert the DAG to one that can be provided to the DECAF model
dag_seed = dag_to_idx(dataset, dag)

def eval_model(dataset_train, dataset_test):
    """Helper function that prints evaluation metrics."""

    X_train, y_train = dataset_train.drop(columns=['income']), dataset_train['income']
    X_test, y_test = dataset_test.drop(columns=['income']), dataset_test['income']

    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred)
    dp = DP(clf, X_test)
    ftu = FTU(clf, X_test)

    return {'precision': precision, 'recall': recall, 'auroc': auroc,
            'dp': dp, 'ftu': ftu}
    
def recursive_parents(dag_seed, child):
    explored=[]
    to_explore=[child]
    parents=set()
    parents.add(child)
    while to_explore:
        child=to_explore.pop()
        for edge in dag_seed:
            if edge[1]==child and edge[0] not in explored:
                explored.append(edge[0])
                parents.add(edge[0])
                to_explore.append(edge[0])
    return parents


def create_bias_dict(df, dag_seed, target, restricted, fairness_type='CF', protected=None):
    """
    Convert the given edge tuples to a bias dict used for generating
    debiased synthetic data.
    """
    targetID= df.columns.get_loc(target)
    restrictedID = df.columns.get_loc(restricted)
    protectedID=[]
    if protected:
        for elem in protected:
            protectedID.append(df.columns.get_loc(elem))
    bias_dict = {}
    if fairness_type=='CF':
        if protected==None:
            return create_bias_dict(df, dag_seed, target, restrictedID, 'DP', protectedID)
        else:
            for edge in dag_seed:
                if edge[1]==targetID:
                    if edge[0] not in protectedID and restrictedID in recursive_parents(dag_seed, edge[0]): #find all parents
                        if edge[1] in bias_dict:
                            bias_dict[edge[1]].append(edge[0])
                        else:
                            bias_dict[edge[1]]=[edge[0]]
    elif fairness_type=='DP':
        # get parents for target
        #get parent parents recursively, add edge if parent parents is restricted
        for edge in dag_seed:
            if edge[1]==targetID:
                if restrictedID in recursive_parents(dag_seed, edge[0]): #find all parents
                    if edge[1] in bias_dict:
                        bias_dict[edge[1]].append(edge[0])
                    else:
                        bias_dict[edge[1]]=[edge[0]]
    elif fairness_type=='FTU':
        bias_dict[targetID] = [df.columns.get_loc(restricted)]
    return bias_dict
    #{0:[1,2,3]} edge 1 into 0 edge 2 into 0 etc

# Bias dictionary to satisfy FTU
bias_dict_cf = create_bias_dict(dataset, dag_seed, 'income', 'sex', fairness_type='CF', 
protected=['occupation', 'hours-per-week', 'workclass', 'education'])
bias_dict_dp = create_bias_dict(dataset, dag_seed, 'income', 'sex', fairness_type='DP', 
protected=None)
bias_dict_ftu = create_bias_dict(dataset, dag_seed, 'income', 'sex', fairness_type='FTU', 
protected=None)

synth_data = train_decaf(dataset_train, dag_seed)
results={}
print('Biased')
# print(eval_model(synth_data, dataset_test))
results['biased']=eval_model(synth_data, dataset_test)
print('DAG SEED A:', dag_seed)

synth_data = train_decaf(dataset_train, dag_seed, biased_edges=bias_dict_ftu)
print('FTU')
results['FTU']=eval_model(synth_data, dataset_test)
print('DAG SEED B:', dag_seed)

synth_data = train_decaf(dataset_train, dag_seed, biased_edges=bias_dict_cf)
print('CF')
results['CF']=eval_model(synth_data, dataset_test)
print('DAG SEED C:', dag_seed)

synth_data = train_decaf(dataset_train, dag_seed, biased_edges=bias_dict_dp)
print('DP')
results['DP']=eval_model(synth_data, dataset_test)
print('DAG SEED D:', dag_seed)
print(results)