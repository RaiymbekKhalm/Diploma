from flask import Flask, request, render_template
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Загрузка обученной модели
model = keras.models.load_model('C:/Users/User/Desktop/diploma/Models/best_model_weights_2.h5')

# Функция для классификации изображения
def classify_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    if class_index == 0:
        return "Бактериальная пневмония"
    elif class_index == 1:
        return "Здоровые"
    elif class_index == 2:
        return "Туберкулез"
    else:
        return "Вирусная пневмония"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("file")
    results = {"Бактериальная пневмония": [], "Туберкулез": [], "Здоровые": [], "Вирусная пневмония": []}

    for file in uploaded_files:
        if file:        
            file.save('uploaded_image.jpg')
            img = image.load_img('uploaded_image.jpg', target_size=(170, 170))
            result = classify_image(img)
            results[result].append(file.filename)

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)