from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_image_data_generators(data_dir, input_shape, batch_size):
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.1)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    return train_generator, validation_generator