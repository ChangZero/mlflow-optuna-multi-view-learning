import yaml
import os
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split

from module.model import vgg16Model, resnet50Model, xceptionModel, nesnatlargeModel, inceptionV3Model, mobileNetModel, cnnModel
from module.util import my_seed, load_df, load_x_data, load_y_data
from module.inference import plot_loss_graph, plot_roc_curve
import warnings
warnings.filterwarnings(action='ignore')

import matplotlib.pyplot as plt


global cfg

def select_model(model_name ,target_size):
    if model_name == "VGG16":
        model = vgg16Model(target_size)
    elif model_name == "InceptionV3":
        model = inceptionV3Model(target_size)
    elif model_name == "ResNet50":
        model = resnet50Model(target_size)
    elif model_name == "MobileNet":
        model = mobileNetModel(target_size)
    elif model_name == "CNN":
        model = cnnModel(target_size)
    else:
        print("Invalid model_name")

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = cfg["lr"]),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = tf.keras.metrics.BinaryAccuracy())
    
    return model

def make_plot(history, save_path, model_name, f_num=None):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='valid loss')
    plt.legend()
    plt.xlabel('Epoch'); plt.ylabel('loss')
    plt.savefig(save_path + "/loss_graph/" + model_name + str(f_num) + ".jpg")
    plt.close()




def train():
    result_dir = datetime.now().strftime("../result_dir/%Y-%m-%d_%H:")
    save_path = os.path.join(result_dir, cfg['model_name'])
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + "/log_file", exist_ok=True)
    os.makedirs(save_path + "/loss_graph", exist_ok=True)
    os.makedirs(save_path + "/h5_model", exist_ok=True)
    os.makedirs(save_path + "/test_result", exist_ok=True)


    tf.keras.backend.clear_session() 
    
    for f_num in range(1,6,1):
        f_list = [1,2,3,4,5]
        f_list.remove(f_num)
        tmp_df1 = load_df(f'{cfg["train_path"]}fold_{str(f_list[0])}.txt')
        tmp_df2 = load_df(f'{cfg["train_path"]}fold_{str(f_list[1])}.txt')
        tmp_df3 = load_df(f'{cfg["train_path"]}fold_{str(f_list[2])}.txt')
        tmp_df4 = load_df(f'{cfg["train_path"]}fold_{str(f_list[3])}.txt')
        train_df = pd.concat([tmp_df1, tmp_df2, tmp_df3, tmp_df4], ignore_index = True)
        # test_df = load_df(cfg["train_path"] + "fold_" + str(cfg['f_num']) + ".txt")
        
        x_t, x_v, y_t, y_v = train_test_split(list(train_df["filename"]), list(train_df["y_label"]), stratify=list(train_df["y_label"]), test_size=0.125)

        print("train data loading")
        x_train = load_x_data(x_t, cfg['method'], cfg['target_size'])
        y_train = load_y_data(y_t)
        
        print("validation data loading")
        x_val = load_x_data(x_v, cfg['method'], cfg['target_size'])
        y_val = load_y_data(y_v)
        
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            class CustomCallback(Callback):
                def on_train_begin(self, logs = None):
                    raw_data = {'epoch' : [],
                                'train_loss' : [],
                                'train_accuracy' : [],
                                'validation_loss' : [],
                                'validation_accuracy': [],
                                }
                    df = pd.DataFrame(raw_data)
                    df.to_csv(save_path + "/log_file/" + cfg["model_name"] + str(f_num) + ".csv", index = False)

                def on_epoch_end(self, epoch, logs=None):
                    df = pd.read_csv(save_path + "/log_file/" + cfg["model_name"] + str(f_num) + ".csv")
                    df.loc[-1]=[epoch, logs["loss"], logs["binary_accuracy"], logs["val_loss"], logs["val_binary_accuracy"]]
                    df.to_csv(save_path + "/log_file/" + cfg["model_name"]  + str(f_num) + ".csv", index = False)

            # filename = (save_path + "/h5_model/" + cfg["model_name"] + str(f_num) + ".h5")
            
            # checkpoint = ModelCheckpoint(filename,
            #                             monitor = 'val_loss',
            #                             verbose = 1,
            #                             save_best_only = True,
            #                             mode = 'auto')

            earlystopping = EarlyStopping(monitor = 'val_loss', 
                                        patience = 20,
                                        )
            
            model = select_model(cfg['model_name'], cfg['target_size'])
            
        history = model.fit(x=(x_train[0], x_train[1], x_train[2]),
                y= y_train, validation_data=((x_val[0], x_val[1], x_val[2]), y_val),
                epochs = cfg['epoch'], batch_size= cfg['batch_size'],
                callbacks = [earlystopping, CustomCallback()])
        make_plot(history, save_path, cfg["model_name"], f_num)
    
def main():
    global cfg
    
    with open("../best-config.yaml", "r") as f:
        cfg = yaml.full_load(f)

    seed = cfg["seed"]
    my_seed(seed)
    train()


    return 0    

if __name__ == '__main__':
    # config = parse_opt()
    main()