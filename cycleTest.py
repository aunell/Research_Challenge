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
    ##can add these edges to make feedback loops
    # ['income', 'education'],
    # ['income', 'occupation'],
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


synth_data = train_decaf(dataset_train, dag_seed, fairness_type=None, restricted='sex', protected=None, target='income', df=dataset)
results={}
results['biased']=eval_model(synth_data, dataset_test)

synth_data = train_decaf(dataset_train, dag_seed, fairness_type='FTU', restricted='sex', protected=None, target='income', df=dataset)
results['FTU']=eval_model(synth_data, dataset_test)
synth_data = train_decaf(dataset_train, dag_seed, fairness_type='CF', restricted='sex', protected=['occupation', 'hours-per-week', 'workclass', 'education'], target='income', df=dataset)
results['CF']=eval_model(synth_data, dataset_test)

synth_data = train_decaf(dataset_train, dag_seed, fairness_type='DP', restricted='sex', protected=None, target='income', df=dataset)
results['DP']=eval_model(synth_data, dataset_test)
print(results)