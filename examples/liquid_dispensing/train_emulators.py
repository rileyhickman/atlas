#!/usr/bin/env python

import pickle

import numpy as np
import olympus
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.models import BayesNeuralNet
from sklearn.metrics import mean_squared_error, r2_score

search_space = {
    "batch_size": hp.quniform("batch_size", 10, 50, 10),
    "hidden_act": hp.choice("hidden_act", ["leaky_relu"]),
    "hidden_depth": hp.quniform("hidden_depth", 2, 5, 1),
    "hidden_nodes": hp.quniform("hidden_nodes", 28, 104, 4),
    "learning_rate": hp.uniform("learning_rate", 1e-5, 5e-3),
    "reg": hp.uniform("reg", 0.001, 1.0),
}
int_params = ["batch_size", "hidden_depth", "hidden_nodes"]


def objective(params):
    # build emualtor
    for param, val in params.items():
        if param in int_params:
            params[param] = int(val)
    model = BayesNeuralNet(
        **params,
        task="regression",
        out_act=dataset_params[current_dataset]["out_act"],
    )
    emulator = Emulator(
        dataset=current_dataset,
        model=model,
        feature_transform=dataset_params[current_dataset]["feature_transform"],
        target_transform=dataset_params[current_dataset]["target_transform"],
    )

    scores = emulator.train()
    loss = scores["test_rmsd"]

    all_losses.append(loss)
    all_cv_scores.append(scores)
    all_params.append(params)
    all_emulators.append(emulator)
    all_test_indices.append(emulator.dataset.test_indices)

    return {"loss": loss, "status": STATUS_OK}


# datasets to emulate

dataset_names = [
    "liquid_ace_100",
    "liquid_hep_100",
    "liquid_thf_100",
    "liquid_thf_500",
]

dataset_params = {
    "liquid_ace_100": {
        "out_act": "relu",
        "feature_transform": "standardize",
        "target_transform": "mean",
    },
    "liquid_hep_100": {
        "out_act": "relu",
        "feature_transform": "standardize",
        "target_transform": "mean",
    },
    "liquid_thf_100": {
        "out_act": "relu",
        "feature_transform": "standardize",
        "target_transform": "mean",
    },
    "liquid_thf_500": {
        "out_act": "relu",
        "feature_transform": "standardize",
        "target_transform": "mean",
    },
}


best_scores = {}

for dataset_name in dataset_names:

    current_dataset = dataset_name
    print("CURRENT DATASET : ", current_dataset)

    all_emulators = []
    all_losses = []
    all_cv_scores = []
    all_params = []
    all_test_indices = []

    trials = Trials()

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=2,
        trials=trials,
    )

    best_idx = np.argmin(all_losses)
    best_emulator = all_emulators[best_idx]

    best_emulator.save(f"emulator_{current_dataset}_BayesNeuralNet")

    # make predictions
    dataset = Dataset(kind=dataset_name)

    train_params = dataset.train_set_features.to_numpy()
    train_values = dataset.train_set_targets.to_numpy()
    test_params = dataset.test_set_features.to_numpy()
    test_values = dataset.test_set_targets.to_numpy()

    train_preds, _, __ = best_emulator.run(train_params, num_samples=50)
    test_preds, _, __ = best_emulator.run(test_params, num_samples=50)

    acc_train_r2 = r2_score(train_values[:, 0], train_preds[:, 0])
    acc_train_rmsd = np.sqrt(
        mean_squared_error(train_values[:, 0], train_preds[:, 0])
    )

    acc_test_r2 = r2_score(test_values[:, 0], test_preds[:, 0])
    acc_test_rmsd = np.sqrt(
        mean_squared_error(test_values[:, 0], test_preds[:, 0])
    )

    std_train_r2 = r2_score(train_values[:, 1], train_preds[:, 1])
    std_train_rmsd = np.sqrt(
        mean_squared_error(train_values[:, 1], train_preds[:, 1])
    )

    std_test_r2 = r2_score(test_values[:, 1], test_preds[:, 1])
    std_test_rmsd = np.sqrt(
        mean_squared_error(test_values[:, 1], test_preds[:, 1])
    )

    best_scores[current_dataset] = {
        "scores": all_cv_scores,
        #'emulators': all_emulators,
        "params": all_params,
        "losses": all_losses,
        "all_test_indices": all_test_indices,
        "train_params": train_params,
        "train_values": train_values,
        "test_params": test_params,
        "test_values": test_values,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "acc_train_r2": acc_train_r2,
        "acc_train_rmsd": acc_train_rmsd,
        "acc_test_r2": acc_test_r2,
        "acc_test_rmsd": acc_test_rmsd,
        "std_train_r2": std_train_r2,
        "std_train_rmsd": std_train_rmsd,
        "std_test_r2": std_test_r2,
        "std_test_rmsd": std_test_rmsd,
    }
    pickle.dump(best_scores, open("results/best_scores.pkl", "wb"))
