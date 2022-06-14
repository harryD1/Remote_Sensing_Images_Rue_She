from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageOps
import streamlit as st

model = load_model('model.h5')

st.write("""
         # Remote Sensing Image Classification Using Deep Convolutional Neural Networks.
         """)
st.write("This is a classification web app to predict remote ssensing images")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
classes = np.array(labels)

def import_and_predict(image_data, model):
        size = (64,64)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC))/255.
        img_reshape = img_resize[np.newaxis,...]
        prediction = model.predict(image.reshape(1,64,64,3))
        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=False)
    prediction = import_and_predict(image, model)
    
    top_3 = np.argsort(prediction[0])[:-4:-1]
    for i in range(3):
        st.write("{}".format(classes[top_3[i]])+" ({:.3})".format(prediction[0][top_3[i]]))

#Keras-Preprocessing
#Version: 1.1.2



