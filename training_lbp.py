#modo de uso, en el cmd se pega el siguiente comando, dentro del folder donde
#se encuentra el archivo: training_lbp.py --training C:\Python27\imagen\dataSet
#se importan las librerias necesarias para el funcionamiento
import numpy as np
from imutils import paths
import cv2
import os
import argparse
from skimage import feature
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
import pickle
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

#se crea la variable que recibe el argumento para el entrenamiento
ap = argparse.ArgumentParser()
#se recibe el argumento que se envia desde la ventana cmd
ap.add_argument("-t", "--training", required=True, help="path to the training images")
#se guarda la direccion de la carpeta que contiene todas las imagenes del dataset
args = vars(ap.parse_args())


#con este metodo se extraen las caracteristicas usando el LBP, recibe la imagen y un factor para normalizar el histograma
def histo_pat(img, eps=1e-7):
    #se extraen las caracteristicas usando LBP, se ingresa la imagen a extraer caracteristicas, el radio, numero de puntos y el metodo.
    lkp = local_binary_pattern(img,p,r,'uniform')     
    
    # se crea el histograma de patrones
    (hist,_) = np.histogram(lkp.ravel(),bins =np.arange(0,  60), range = (0, 254))

    #normalizar histrograma de patrones
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


#se crean los valores de r (radio) y p (numero de pixeles vecinos a comparar)
r = 2  
p = 8

#se declaran los vectores de data y labels que seran llenados para el entrenamiento del SVM
data = [] 
labels = []

#se llama el haarcascade para detectar la cara de la imagen y poder deshacernos del fondo
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')





#metodo para leer las imagenes del daraset, recibe el argumento que contiene la direccion de las imagenes
for imagePath in paths.list_images(args["training"]):
    #se lee una imagen del directorio ingresado
    imag = cv2.imread(imagePath)
    #se convierte a escala de grises para poder operarla
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    #se aplica filtro bilateral para eliminar ruido
    suave = cv2.bilateralFilter(gray,7,75,75)
    #se detecta el rectangulo que encierra la cara con el haarcascade
    faces = faceCascade.detectMultiScale(suave, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))
    #el siguiente metodo es para recortar la imagen y extraer caracteristicas
    for i in faces:
        #se incializa y se reinicia grid para guardar las features de una sola cara
        grid=[]
        #se recorta la cara de la imagen, para esto se utilizan las coordenadas del rectangulo que retorna el haarcascade
        gy_1= suave[i[1]:(i[1]+i[3]),i[0]+10:(i[0]+i[2]-10)]
        #se hace un resize de la imagen para dejar un tamano estandar
        gy =cv2.resize(gy_1, (130, 151))
        #se ecualiza el histograma para poder mejorar el contraste de la imagen
        const = cv2.equalizeHist(gy)
        print imagePath
        #se inicia un ciclo for para dividir la imagen en 7x7 segmentos
        for j in range(7):
            
            for k in range(7):
                #se realizan los recortes de cada seccion dejando pequeñas
                #imagenes de 18x21
                gy_rec=const[18*k:(18*(k+1)),21*j:(21*(j+1))]
                #se calcula el histograma de LBP de cada pequeño segmento
                hist=histo_pat(gy_rec)
                #se agrega el histograma a grid, al final queda con cerca
                #de 2891 features, 7x7x59bin
                grid.extend(hist)


        
        #se extrae el nombre de la carpeta en cuestion, ese es el nombre de la persona. y se guarda en labels
        labels.append(os.path.split(os.path.dirname(imagePath))[-1])
        #se guarda el histograma de patrones de cada persona
        data.append(grid)

    

#se crea la maquina de soporte vectorial lineal
svm = LinearSVC(C=70.0)
ml = CalibratedClassifierCV(svm)
#se entrena la maquina de soporte vectorial con el metodo fit()
ml.fit(data, labels)
#se guarda la maquina entrenada como un archivo para ser llamado en el programa de testeo
filename = 'finalized_SVM_NF.sav'
pickle.dump(ml, open(filename, 'wb'))
#se imprime DONE para saber que ya ha terminado
print("[DONE]")
    

