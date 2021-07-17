import keras
from keras import Input
from keras.layers.experimental.preprocessing import RandomFlip,RandomRotation,Rescaling
from CONSTANTS import image_size,batch_size
from get_data import get_dataset
import matplotlib.pyplot as plt



def preprocessing(input_shape):

    inputs = Input(shape=input_shape)
    data_augmentation = keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.5),
        Rescaling(0.5)
    ])
    x=data_augmentation(inputs)
    # standardization
    x = Rescaling(1.0 / 255)(x)

    return x,inputs

def data_viz(train_ds):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
            plt.show()

def data_aug_viz(data_augmentation,train_ds):

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
            plt.show()


if __name__ == '__main__':

    preprocessing(image_size + (3,))
    train_ds, val_ds= get_dataset(image_size,batch_size)

