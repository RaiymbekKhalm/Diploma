import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model 

# Путь к директории с данными
data_dir = "C:/Users/User/Desktop/diploma/Dataset/wholeset/train"
test_dir = "C:/Users/User/Desktop/diploma/Dataset/wholeset/test"

# Путь для сохранения логов TensorBoard
log_dir = "C:/Users/User/Desktop/diploma/logs/final_model"
#tensorboard --logdir=C:/Users/User/Desktop/diploma/logs/log_Low_model
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Создание генератора изображений
datagen = ImageDataGenerator(
    rescale=1.0/255, 
    validation_split=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Загрузка и подготовка данных для обучения и валидации
input_shape = (170, 170, 3)
batch_size = 100

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

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size= input_shape[:2],
    batch_size= batch_size,
    class_mode='categorical'
)

# Создание колбека ModelCheckpoint с мониторингом точности на валидационном наборе
checkpoint_path = "C:/Users/User/Desktop/diploma/Models/best_model_weights_2.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor='val_accuracy',
    save_best_only=True,
    mode='max', 
    verbose=1
)



# Создание модели нейронной сети
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Компиляция модели
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.001), 
    metrics=['accuracy']
)

# Обучение модели
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=35,
    callbacks=[tensorboard, checkpoint]
)

# Сохранение модели
model.save('C:/Users/User/Desktop/diploma/Models/final_model.h5')

# Оценка модели на тестовых данных
eval_result = model.evaluate(test_generator)

# Вывод процента успешных предсказаний
accuracy_percentage = eval_result[1] * 100
print(f"Accuracy on test data: {accuracy_percentage:.2f}%")

# Загрузка модели с сохраненными весами
loaded_model = load_model(checkpoint_path)

# Оценка модели на тестовых данных
test_loss, test_accuracy = loaded_model.evaluate(test_generator)
print(f'Best weights: Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

model.summary()