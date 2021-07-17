import os
from keras.models import load_model
from flask import Flask

from CONSTANTS import image_size,batch_size
from config import model_path,test_folder

from predict import predict
from get_data import get_dataset
from get_model import build_model,train


app = Flask(__name__)

@app.route('/train_model', methods=['GET'])
def train_model():
    train_ds, val_ds = get_dataset(image_size, batch_size)
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    model = build_model(image_size + (3,))
    print(model.summary())
    train(train_ds, val_ds,model)
    return "training completed"


@app.route('/predict_img', methods=['GET'])
def predict_img():
    dict = {}
    model = load_model(model_path)
    for filename in os.listdir(test_folder):
        result=predict(model,os.path.join(test_folder,filename))
        dict[filename] = result

    return dict


if __name__ == '__main__':
    app.run(debug=True)







