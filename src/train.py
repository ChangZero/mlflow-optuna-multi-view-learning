import yaml
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

import mlflow
from optuna.integration.mlflow import MLflowCallback

from module.model import vgg16Model, resnet50Model, xceptionModel, nesnatlargeModel, inceptionV3Model, mobileNetModel, cnnModel
from module.util import load_df, load_x_data, load_y_data
from module.inference import plot_loss_graph, plot_roc_curve
import warnings
warnings.filterwarnings(action='ignore')

TARGET_SIZE = [400, 400, 1]
global cfg
global x_train
global y_train
    
mlflc = MLflowCallback(
    # tracking_uri="http://127.0.0.1:5000",
    metric_name="metric",
)

def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-4, log=True
        )
        kwargs["weight_decay"] = trial.suggest_float("rmsprop_weight_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-3, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-4, log=True)
        kwargs["weight_decay"] = trial.suggest_float("adam_weight_decay", 0.85, 0.99)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-4, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-3, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer

def create_model(optimizer, model_name, target_size=TARGET_SIZE):
    
    if model_name == "VGG16":
        model = vgg16Model(target_size)
    elif model_name == "InceptionV3":
        model = inceptionV3Model(target_size)
    elif model_name == "ResNet50":
        model = resnet50Model(target_size)
    elif model_name == 'CNN':
        model = cnnModel(target_size)
    elif model_name == 'Xception':
        model = xceptionModel(target_size)
    elif model_name == 'Nesnatlarge':
        model = nesnatlargeModel(target_size)
    elif model_name == 'Mobile':
        model = mobileNetModel(target_size)
    else:
        print("Invalid model_name")

    # Compile model.
    model.compile(
        optimizer= optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['acc'],
        # metrics=['acc', f1_m, precision_m, recall_m],
    )

    return model

@mlflc.track_in_mlflow()
def objective(trial):
    global cfg
    global x_train
    global y_train
    
    tf.keras.backend.clear_session()
    # os.makedirs(model_save_path + f"/plot/{str(fold_number)}", exist_ok=True)
    optimizer = create_optimizer(trial)
        
    model = create_model(optimizer, cfg['model_name'], target_size=TARGET_SIZE)
    
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 10)
    pruning = TFKerasPruningCallback(trial, monitor='val_loss')
    history = model.fit(x=(x_train[0], x_train[1], x_train[2]),
              y= y_train, validation_split=0.2,
              epochs = cfg['epoch'], batch_size= cfg['batch_size'],
              callbacks = [earlystopping, pruning])
    loss, accuracy= model.evaluate(x=(x_train[0], x_train[1], x_train[2]), y=y_train)

    plot_loss_graph(history)
    mlflow.log_artifacts("../plot")
    
    return accuracy

def main():
    global cfg
    global x_train
    global y_train
    
    methods = ["origin", "median_blur", "sobel_masking"]
    with open("../config.yaml", "r") as f:
        cfg = yaml.full_load(f)
        
    train_path = cfg["train_path"]
    # epoch = cfg["epoch"]
    # batch_size = data["batch_size"]
    # model_name = data["model_name"]
    seed = cfg["seed"]
    
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_df = load_df(train_path)
    
    x_train = load_x_data(train_path, methods, target_size=TARGET_SIZE, df = train_df)
    
    y_train = load_y_data(train_df)
    
    mlflow.autolog()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed), pruner=SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=300, timeout=10000000)
    
    return 0    

if __name__ == '__main__':
    # config = parse_opt()
    main()