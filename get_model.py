from keras import Sequential
from keras.callbacks import ModelCheckpoint
from CONSTANTS import image_size,batch_size
from config import epochs
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Dense,Dropout,Flatten
from get_data import get_dataset
from predict import predict


def build_model(dim):
    (Image_Width, Image_Height, Image_Channels) =dim

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(Image_Width, Image_Height, Image_Channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    return model


def train(train_ds,val_ds,model,epochs=epochs):

    callbacks = [
        ModelCheckpoint("save_at_{epoch}.h5"),
    ]

    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

    return 'trained'


if __name__ == '__main__':
    train_ds,val_ds= get_dataset(image_size,batch_size)
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    model = build_model(image_size + (3,))
    print(model.summary())
    train(train_ds,val_ds,epochs=5)
    score=predict(model)