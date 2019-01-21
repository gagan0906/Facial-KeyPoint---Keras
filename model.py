from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization
from keras.models import Sequential

def create_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, input_shape=(150, 150, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=16, kernel_size=3))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=32, kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))


    model.add(Dense(136))


    # Summarize the model
    return model  