import yaml
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

import mlflow
from optuna.integration.mlflow import MLflowCallback

from module.model import vgg16Model, resnet50Model, xceptionModel, nesnatlargeModel, inceptionV3Model, mobileNetModel, cnnModel
from module.util import load_df, load_x_data, load_y_data, load_x_df, load_y_df
from module.inference import plot_loss_graph, plot_roc_curve, export_test_result, export_hit_eval_result, export_eval_result


import warnings
warnings.filterwarnings(action='ignore')

# TARGET_SIZE = [400, 400, 1]
global cfg
global train_path
    
mlflc = MLflowCallback(
    # tracking_uri="http://127.0.0.1:5000",
    metric_name="metric",
)

def create_optimizer(trial):
    global cfg
    
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    # optimizer_options = ["RMSprop", "Adam", "SGD"]
    # optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    optimizer_selected = "Adam"
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

def create_model(optimizer, model_name, target_size):
    target_size = cfg['target_size']
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

def objective(trial, train_df, test_df, f_num):
    global cfg
    tf.keras.backend.clear_session()
    
    x_t, x_v, y_t, y_v = train_test_split(list(train_df["filename"]), list(train_df["y_label"]), stratify=list(train_df["y_label"]), test_size=0.125)

    print("train data loading")
    x_train = load_x_data(x_t, cfg['method'], cfg['target_size'])
    y_train = load_y_data(y_t)
    
    print("validation data loading")
    x_val = load_x_data(x_v, cfg['method'], cfg['target_size'])
    y_val = load_y_data(y_v)
    
    # Multi GPU
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        optimizer = create_optimizer(trial)    
        model = create_model(optimizer, cfg['model_name'], target_size=cfg['target_size'])
    
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 10)
    pruning = TFKerasPruningCallback(trial, monitor='val_loss')
    history = model.fit(x=(x_train[0], x_train[1], x_train[2]),
              y= y_train, validation_data=((x_val[0], x_val[1], x_val[2]), y_val),
              epochs = cfg['epoch'], batch_size= cfg['batch_size'],
              callbacks = [earlystopping, pruning])

    val_loss, val_acc = model.evaluate(x=(x_val[0], x_val[1], x_val[2]), y=y_val)
    # model.save(f'../weights/model_{f_num}.h5')
    mlflow.log_metric(f"val_loss_{f_num}", val_loss)
    mlflow.log_metric(f"val_acc_{f_num}", val_acc)
    plot_loss_graph(history, f_num)

    # export_hit_eval_result(df, f_num)

    mlflow.log_artifacts("../plot")
    mlflow.log_artifacts("../weights")
    
    return np.min(history.history["val_loss"])

@mlflc.track_in_mlflow()
def objective_cv(trial):
    global cfg
    
    k_scores = []
    for f_num in range(1,6,1):
        f_list = [1,2,3,4,5]
        f_list.remove(f_num)
        tmp_df1 = load_df(f'{cfg["train_path"]}fold_{str(f_list[0])}.txt')
        tmp_df2 = load_df(f'{cfg["train_path"]}fold_{str(f_list[1])}.txt')
        tmp_df3 = load_df(f'{cfg["train_path"]}fold_{str(f_list[2])}.txt')
        tmp_df4 = load_df(f'{cfg["train_path"]}fold_{str(f_list[3])}.txt')
        train_df = pd.concat([tmp_df1, tmp_df2, tmp_df3, tmp_df4], ignore_index = True)
        test_df = load_df(cfg["train_path"] + "fold_" + str(f_num) + ".txt")
        
        loss = objective(trial, train_df, test_df, f_num)
        k_scores.append(loss)
    return np.mean(k_scores)

def main():
    global cfg
    
    with open("../config.yaml", "r") as f:
        cfg = yaml.full_load(f)

    seed = cfg["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    mlflow.autolog()
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=SuccessiveHalvingPruner())
    study.optimize(objective_cv, n_trials=300, timeout=None)
    
    return 0    

if __name__ == '__main__':
    # config = parse_opt()
    main()