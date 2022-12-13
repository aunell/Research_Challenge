"""Module containing training functions for the various models evaluated in the DECAF paper."""

import os
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tensorflow as tf
import torch
from sklearn.neural_network import MLPClassifier
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import WGAN_GP, VanilllaGAN
import networkx as nx

from data import DataModule
from models.DECAF import DECAF

models_dir = 'cache'
def all_parents(dag_seed, child):
    '''
    find all nodes that are in the parental lineage the child, including parents of parents, etc.
    inclusive of child node
    '''
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

def create_bias_dict(df, dag_seed, target, restricted, fairness_type='FTU', protected=None):
    """
    Convert the given edge tuples to a bias dict used for generating
    debiased synthetic data.
    """
    if not fairness_type:
        return {}
    targetID= df.columns.get_loc(target)
    restrictedID = df.columns.get_loc(restricted)
    protectedID=[]
    bias_dict = {targetID: []}
    if protected and fairness_type=='CF':
        for elem in protected:
            protectedID.append(df.columns.get_loc(elem))
    if fairness_type=='FTU':
        bias_dict[targetID] = [df.columns.get_loc(restricted)]
    else:
        for edge in dag_seed:
            if edge[1]==targetID:
                if edge[0] not in protectedID and restrictedID in all_parents(dag_seed, edge[0]): #find all parents
                    bias_dict[edge[1]].append(edge[0]) #can assume edge[0] not in protectedID if protectedID should be empty if protected is nothing bc of DP
    # print('🟡 bias dictionary',bias_dict)
    return bias_dict


def train_decaf(train_dataset, dag_seed, fairness_type=None, h_dim=200, lr=0.5e-3,
                batch_size=64, lambda_privacy=0, lambda_gp=10, d_updates=10,
                alpha=2, rho=2, weight_decay=1e-2, grad_dag_loss=False, l1_g=0,
                l1_W=1e-4, p_gen=-1, use_mask=True, epochs=50, model_name='decaf', 
                restricted=None, protected=None, target=None, df=None):
    model_filename = os.path.join(models_dir, f'{model_name}.pkl')
    
    dm = DataModule(train_dataset.values)

    model = DECAF(
        dm.dims[0],
        dag_seed=dag_seed,
        h_dim=h_dim,
        lr=lr,
        batch_size=batch_size,
        lambda_privacy=lambda_privacy,
        lambda_gp=lambda_gp,
        d_updates=d_updates,
        alpha=alpha,
        rho=rho,
        weight_decay=weight_decay,
        grad_dag_loss=grad_dag_loss,
        l1_g=l1_g,
        l1_W=l1_W,
        p_gen=p_gen,
        use_mask=use_mask,
    )

    # print('🔴 dag seed before untangling', dag_seed)
    dag_seed_untangeled=model.detangle_graph()
    # print('🟠 dag seed untangled', dag_seed_untangeled)
    trainer = pl.Trainer(max_epochs=epochs, logger=False)
    trainer.fit(model, dm) #DOESNT WORK WITH CYCLIC IN ORIGINAL IMPLEMENTATION
    torch.save(model, model_filename)
    biased_edges = create_bias_dict(df, dag_seed_untangeled, target, restricted, fairness_type, protected)

    # Generate synthetic data
    synth_dataset = (
        model.gen_synthetic(
            dm.dataset.x,
            gen_order=model.get_gen_order(),
            biased_edges=biased_edges,
        )
        .detach()
        .numpy()
    )
    synth_dataset[:, -1] = synth_dataset[:, -1].astype(np.int8)

    synth_dataset = pd.DataFrame(synth_dataset,
                                 index=train_dataset.index,
                                 columns=train_dataset.columns)

    if 'approved' in synth_dataset.columns:
        # Binarise columns for credit dataset
        synth_dataset['ethnicity'] = np.round(synth_dataset['ethnicity'])
        synth_dataset['approved'] = np.round(synth_dataset['approved'])
    else:
        # Binarise columns for adult dataset
        synth_dataset['sex'] = np.round(synth_dataset['sex'])
        synth_dataset['income'] = np.round(synth_dataset['income'])

    return synth_dataset
