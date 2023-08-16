import yaml
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import mlflow.keras
from module.util import load_df, load_x_data, load_y_data
from module.inference import export_test_result, export_hit_eval_result, export_excel, export_eval_result

TARGET_SIZE = [512, 512, 1]

mlflow.tensorflow.autolog()

def test(test_path, model_path, model_name):

    result_dir = datetime.now().strftime("../result_dir/%Y-%m-%d_%H:%M")
    result_path = os.path.join(result_dir, model_name)
    
    os.makedirs(result_path, exist_ok=True)
    
    methods = ["origin", "median_blur", "sobel_masking"]
    test_df = load_df(test_path)
    
    x_test = load_x_data(test_path, methods, target_size=TARGET_SIZE, df = test_df)
    y_test = load_y_data(test_df)
    
    
    # test_df = load_df1(test_path)
    
    
    model = mlflow.keras.load_model(model_path)
    
    # acc, loss = model.evaluate(x_test, y_test)
    
    y_prob = model.predict(x=(x_test[0], x_test[1], x_test[2]))
    y_pred = (y_prob > 0.5).astype("int32")
    # print(y_pred)
    df = export_test_result(test_df, y_pred, y_prob, result_path)
    export_eval_result(y_test, y_pred, result_path)
    export_hit_eval_result(df, y_prob, result_path, model_name)
    
    
    # result_dic = {}
    # result_dic[model_name] = result_path
    # model_list = [model_name]
    # export_excel(model_list, result_dic)
        
    
def main():
    with open("../test-config.yaml", "r") as f:
        data = yaml.full_load(f)
        
    test_path = data["test_path"]
    model_path = data["model_path"]
    model_name = data["model_name"]
    # dir_path = data["dir_path"]
    seed = data["seed"]
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    test(test_path, model_path, model_name)

    return 0

if __name__ == '__main__':
    # config = parse_opt()
    main()
    


