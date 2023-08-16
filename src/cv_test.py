import yaml
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import mlflow.keras
from module.util import load_df, load_x_data, load_y_data, load_x_df, load_y_df
from module.inference import export_test_result, export_hit_eval_result, export_excel, export_eval_result

# TARGET_SIZE = [512, 512, 1]

global cfg

def test():
    global cfg
    
    
    result_dir = cfg['result_dir']
    
    for f_num in range(1,6,1):
        f_list = [1,2,3,4,5]
        f_list.remove(f_num)
    
        test_df = load_df(f'{cfg["data_path"]}fold_{f_num}.txt')

        x_test = load_x_df(test_df, cfg["method"], cfg["target_size"])
        y_test = load_y_df(test_df)
        
        # model = mlflow.keras.load_model(cfg['model_path'])
        model = tf.keras.models.load_model(f"{cfg['model_path']}/{cfg['model_name']}{f_num}.h5")
    
        y_prob = model.predict(x=(x_test[0], x_test[1], x_test[2]))
        y_pred = (y_prob > 0.5).astype("int32")
        # print(y_pred)
        
        df = export_test_result(test_df, y_pred, y_prob, result_dir, f_num=f_num)
        export_eval_result(y_test, y_pred, result_dir, f_num=f_num)
        export_hit_eval_result(df,result_dir, f_num=f_num)
    


def main():
    global cfg
    
    with open("../test-config.yaml", "r") as f:
        cfg = yaml.full_load(f)
        
    seed = cfg["seed"]
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    test()
    return 0

if __name__ == '__main__':
    # config = parse_opt()
    main()
    

