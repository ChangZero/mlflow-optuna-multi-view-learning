import os
import random
import numpy as np
import pandas as pd
import cv2
import imgaug.augmenters as iaa
from PIL import Image
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm

def my_seed(seed):
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed) # random
    np.random.seed(seed) # np
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    tf.random.set_seed(seed) # tensorflow
    

def median_blur(image_path, target_size):
    src = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    medianblur = cv2.medianBlur(src, ksize = 3).astype("uint8")
    medianblur = medianblur/255.0
    return medianblur

def noise_drop(image_path, target_size):
    src = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    aug = iaa.Dropout(p=(0, 0.2))(images = src).astype("uint8")
    aug = aug/255.0
    return aug

def his_equalized(image_path, target_size):
    src = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    hist = cv2.equalizeHist(src)
    hist = hist/255.0
    return hist

def sobel_masking_y(image_path, target_size):
    src = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    sobel_y = cv2.Sobel(src, -1, 0, 1, delta=128).astype("uint8")
    sobel_y = sobel_y/255.0
    return sobel_y

def original_image(image_path, target_size):
    src = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    # src = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)
    src = src/255.0
    return src

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# def load_df(file_path):
#     label_name = ["ACC", "REJ"]
#     filelist = []
#     categories = []
#     acc_count = 0
#     for label in label_name:
#         filenames = os.listdir(os.path.join(file_path, label))
#         for filename in filenames:
#             if label == 'ACC':
#                 if acc_count == 198:
#                     continue
#                 categories.append(0)
#                 acc_count += 1
#             else:
#                 categories.append(1)
#             filelist.append(filename)
#     df = pd.DataFrame({
#         'filename': filelist,
#         'y_label': categories
#     })
#     return df

def load_df(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['filename','y_label']
    return df

def load_x_df(df, methods, target_size):
    x_data = []
    # y_data = []
    for method in methods:
        b_x_data = []
        for img, _ in tqdm(zip(df['filename'], df['y_label']), desc="Data Loading", mininterval=0.01, ascii = ' ='):

            if method == "median_blur":
                data = median_blur(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "noise_drop":
                data = noise_drop(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "his_equalized":
                data = his_equalized(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "sobel_masking":
                data = sobel_masking_y(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "origin":
                data = original_image(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            else:
                raise Exception("Invalid method")
    
            b_x_data.append(data)
        # y_data.append(category)
        x_data.append(b_x_data)
        
    x_data = np.array(x_data, dtype=np.float32)
    # y_data = np.array(y_data)
    return x_data


def load_x_data(data, methods, target_size):
    x_data = []
    # y_data = []
    for method in methods:
        b_x_data = []
        # for img in tqdm(data, desc="Data Loading", mininterval=0.01, ascii = ' ='):    
        for img_path in tqdm(data, desc="Data Loading", mininterval=0.01, ascii = ' ='):
            if method == "median_blur":
                img = median_blur(img_path, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "noise_drop":
                img = noise_drop(img_path, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "his_equalized":
                img = his_equalized(img_path, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "sobel_masking":
                img = sobel_masking_y(img_path, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "origin":
                img = original_image(img_path, target_size).reshape(target_size[0], target_size[1], target_size[2])
            else:
                raise Exception("Invalid method")
    
            b_x_data.append(img)
        # y_data.append(category)
        x_data.append(b_x_data)
        
    x_data = np.array(x_data, dtype=np.float32)
    # y_data = np.array(y_data)
    return x_data

def load_y_df(df):
    y_data = np.array(df['y_label'], dtype=np.float32)
    return y_data

def load_y_data(data):
    y_data = np.array(data, dtype=np.float32)
    return y_data


# prototype code testing function
def load_df2(file_path):
    label_name = ["ACC", "REJ"]
    filelist = []
    categories = []
    acc_count = 0
    rej_count = 0
    for label in label_name:
        filenames = os.listdir(os.path.join(file_path, label))
        for filename in filenames:
            if acc_count + rej_count == 40:
                break
            if label == 'ACC':
                if acc_count == 20:
                    continue
                categories.append(0)
                acc_count += 1
            else:
                if rej_count == 20:
                    continue
                categories.append(1)
                rej_count += 1
            filelist.append(filename)
    test_df = pd.DataFrame({
        'filename': filelist,
        'y_label': categories
    })

    return test_df


def get_weld_image(img_path, target_size):

    img = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY), (1256, 1256))
    image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image = cv2.resize(image, (512, 512))
    image = np.array(image, dtype=np.float32)
    image = cv2.GaussianBlur(image, (31,31), 0)

    for i in range(9):
        image = np.split(image, 2, axis=0)
        image = np.add(image[0], image[1])
        
    image = np.gradient(np.squeeze(image))
    y1, y2 = 1256 - int(np.argmax(image)* 1256 / target_size[0]), 1256 - int(np.argmin(image)* 1256 / target_size[1]) 

    if y1 < 1256*0.05 or y2 > 1256*0.95:
        raise Exception("y1 or y2 is out of range")

    weld = (y1 - 30, y2 + 30)

    result = img[weld[0] - 30 : weld[1] + 30]

    image_padd = img[: , : y2 + 30]
    image_padd = cv2.rotate(image_padd, cv2.ROTATE_90_CLOCKWISE)
    image_padd = cv2.normalize(image_padd, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    avg = int(np.mean(image_padd))
    norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    h, w = norm.shape
    width,height = 1256, 1256
    dst = avg * np.ones((height, width), dtype=np.uint8)
    roi = result[0:h, 0:w]
    dst[int(628-h/2):int(628+h/2), 0:1256] = roi

    return dst