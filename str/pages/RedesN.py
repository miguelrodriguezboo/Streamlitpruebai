import streamlit as st

import matplotlib.pyplot as plt
from keras import Sequential
from keras.datasets import mnist  # En este módulo está MNIST en formato numpy
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import paquetes.modulo as md

#información sobre optimizadores
#https://prompt.uno/redes-neuronales-profundas/algoritmos-de-optimizacion-para-redes-neuronales-profundas/#:~:text=Tipos%20de%20algoritmos%20de%20optimizaci%C3%B3n%20m%C3%A1s%20utilizados%201,ampliamente%20en%20el%20entrenamiento%20de%20redes%20neuronales.%20
def escoger_optimizador(optimizador):
    if optimizador == "Adam":
        return Adam()
    if optimizador == "RMSprop":
        return RMSprop()
    if optimizador == "Adagrad":
        return Adagrad()
    if optimizador == "SGD":
        return SGD()
    if optimizador == "GD":
        return SGD(momentum=0.0)
    return Adam()

def selectores():
    row = st.columns(2)
    row2 = st.columns(2)

    t1 = row[0].container(height=70, border= False)
    epoch = t1.number_input("Selecciona el número de epochs:",min_value=1,step=1)

    t2 = row[1].container(height=70, border= False)
    batch_size = t2.number_input("Selecciona el tamaño de lotes:",min_value=1,step=1)

    row3 = st.columns(2)
    t31 = row3[0].container(height=70, border= False)
    tipo_red = t31.selectbox(
        "Selecciona el tipo de Red Neuronal a utilizar",
        options = ("None","FCN","CNN"),
        index=None,
        placeholder="Red Neuronal",
    )

    t32 = row3[1].container(height=70, border= False)
    tipo_opti = t32.selectbox(
        "Selecciona el optimizador a utilizar",
        options = ("None","Adam","RMSprop","Adagrad","SGD","GD"),
        index=None,
        placeholder="Optimizador",
    )
    return epoch,batch_size,tipo_red,tipo_opti
  
def figura_resultados(modelo, arguments):
    plt.style.use('ggplot')

    fig = plt.figure()
    epoch_values = list(range(arguments['epochs']))

    plt.plot(epoch_values, modelo.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(epoch_values, modelo.history['val_loss'], label='Pérdida de validación')
    plt.plot(epoch_values, modelo.history['accuracy'], label='Exactitud de entrenamiento')
    plt.plot(epoch_values, modelo.history['val_accuracy'], label='Exactitud de validación')
 
    plt.title('Pérdida y Exactitud de Entrenamiento')
    plt.xlabel('Epoch N°')
    plt.ylabel('Pérdida/Exactitud')
    plt.legend()

    return fig

@st.cache_data
def obtener_mnist():
    return mnist.load_data()

md.menu()
arguments = {'epochs':100,
             'batch_size':128,
             'output': "\\image\\grafico.png"}

arguments['epochs'],arguments['batch_size'],tipo_red, tipo_opti = selectores()

if st.button("Entrenar RED"):
    (X_train, y_train), (X_test, y_test) = obtener_mnist()

    if tipo_red == "FCN" or tipo_red == "None":
        X_train = X_train.reshape((X_train.shape[0], 28 * 28 * 1))
        X_test = X_test.reshape((X_test.shape[0], 28 * 28 * 1))
    elif tipo_red == "CNN":
        X_train = X_train.reshape(60000,28,28,1)
        X_test = X_test.reshape(10000,28,28,1)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    model = Sequential()
    if tipo_red == "FCN" or tipo_red == "None":
        model.add(Dense(256, input_shape=(28 * 28 * 1,), activation='sigmoid'))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(10, activation='softmax'))
    elif tipo_red == "CNN":
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

    
    optimizador = escoger_optimizador(tipo_opti)
    model.compile(loss='categorical_crossentropy', optimizer=optimizador, metrics=['accuracy'])

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

    with st.spinner('Entrenando la red, podría llevar unos minutos...'):
        H = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=arguments['epochs'],
              batch_size=arguments['batch_size'])
        predictions = model.predict(X_test, batch_size=64)
        

    st.success('Red entrenada', icon="✅")

    st.write(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in label_binarizer.classes_]))
    
    st.plotly_chart(figura_resultados(H, arguments))