from keras.preprocessing.image import load_img,img_to_array
import tensorflow as tf
from CONSTANTS import  image_size
from config import test_folder,model_path
from keras.models import load_model
import os

def predict(model,img_path):
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0]

    if score[0] >= 0.1 and score[1] < 0.1:
        result="cat found"
    elif score[1] >= 0.1 and score[0] < 0.1:
        result = "dog found"
    elif score[0] >= 0.1 and score[1] >= 0.1:
        result = "dog and cat found"

    return result

if __name__ == '__main__':

    model = load_model(model_path)
    for filename in os.listdir(test_folder):
        result=predict(model,os.path.join(test_folder,filename))