from tensorflow.keras.preprocessing import image_dataset_from_directory
from CONSTANTS import image_size,batch_size
from config import validation_split

def get_dataset(image_size,batch_size):

    train_ds = image_dataset_from_directory(
        "Data/Train_Data/",
        label_mode='categorical',
        validation_split=validation_split,
        subset="training",
        seed=1337,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = image_dataset_from_directory(
        "Data/Train_Data/",
        validation_split=validation_split,
        subset="validation",
        seed=1337,
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size,
    )
    return train_ds,val_ds


if __name__ == '__main__':
    get_dataset(image_size,batch_size)
