#se importan las librerias necesarias para su uso
from skimage.transform import rotate
from skimage import feature
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
from sklearn.svm import LinearSVC
import numpy as np
from imutils import paths
import cv2
import os
import argparse
import sklearn
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import openpyxl
import pickle

#se inician los parametros de radios y numero de pixeles vecinos para el LBP
r = 2  
p = 8

#se crea el haarcascade que nos ayuda a detectar las caras para poder recortarlas de la imagen
#y dejarlas sin fondo
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#se llama el excel notas.xlsx
notas = openpyxl.load_workbook('notas.xlsx')
#se extrae la hoja1 del excel, esta contiene los nombres y las notas de los estudiantes
hoja = notas.get_sheet_by_name('Hoja1')
#se llama el numero maximo de filas de la hoja de excel, con esto sabremos hasta donde comparar nombres
m = hoja.max_row

#metodo para extraer caracteristicas usando LBP
def histo_pat(img, eps=1e-7):
    #se extraen las caracteristicas usando LBP, se ingresa la imagen a extraer caracteristicas, el radio, numero de puntos y el metodo.
    lkp = local_binary_pattern(img,p,r,'uniform')     
    
    #se crea el histograma de patrones
    (hist,_) = np.histogram(lkp.ravel(),bins =np.arange(0,  60), range = (0, 254))

    #normalizar histrograma de patrones
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist

#se carga la maquina de soporte vectorial previamente entrenada, de esta manera podemos hacer predicciones
filename='finalized_SVM_NF.sav'
load_svm = pickle.load(open(filename, 'rb'))


##############################TESTING#######################
#se prende la camara del puerto 0 (webcam)
video_capture = cv2.VideoCapture(0)# 

while True:
    
    #se captura cada frame como una imagen
    ret, frame = video_capture.read()
    #se rota la imagen, esto es porque se tiene el efecto espejo
    frame=cv2.flip(frame,1,0)
    #se convierte la imagen a escala de grises para realizar operaciones sobre esta
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #se aplica filtro bilateral para eliminar el ruido que tenga la imagen
    suave = cv2.bilateralFilter(gray,7,75,75)
    #se detecta la cara en la imagen con el haarcascade que nos retorna las coordenadas del rectangulo que encierra la cara
    faces = faceCascade.detectMultiScale(suave, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))
    #cortar las caras:    
    for i in faces:
        #se incializa y se reinicia grid para guardar las features de una sola cara
        grid=[]
        cint=1
        #se crea la variable que se imprime en la pantalla, esta contiene el nombre del estudiante y las notas
        imp = ''
        #se recorta la cara de la imagen usando las coordenadas del haarcascade, de esta manera no nos importa el fondo
        gy_1= suave[i[1]:(i[1]+i[3]),i[0]+10:(i[0]+i[2]-10)]
        #se hace un resize ya que la SVM fue entrenada con imagenes del mismo tamano
        gy =cv2.resize(gy_1, (130, 151))
        #se ecualiza el histograma para mejorar el contraste
        const = cv2.equalizeHist(gy)
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
                        
        #se realiza la prediccion con los features obtenidos de la rejilla
        prediction = load_svm.predict(grid)[0]
        #se vuelve string para poder imprimirlo
        prediction1 = str(prediction)
        #se inicia un ciclo for para recorrer la primera columna
        for fila in hoja.columns:
            #se pasa por cada fila
            for f in fila:
                #se compara que no se vaya a pasar del maximo de filas usadas
                if cint < m:
                    #se compara cada fila con el nombre obtenido
                    if prediction1 == f.value:
                        for t in hoja[cint]:
                            #se extraen los datos de la fila que encaje con la
                            #prediccion
                            imp = imp + str(t.value) + ' '
                                                                
                cint = cint+1

        

        #se dibuja un rectangulo sobre la camara en la pantalla, se emplean la coordenadas del haarcascade     
        cv2.rectangle(frame, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0, 0, 255), 2)
        #se coloca un texto debajo del rectangulo sobre la cara, este texto contiene el nombre de la persona y las notas.
        cv2.putText(frame,prediction1+prob, (i[0],i[1]+i[3]+20), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255), 2)    
    
    
    
    #se muestra cada frame tomado con la webcam
    cv2.imshow('Video', frame)
    #se cierra la camara en caso que se presione la letra "q" minuscula
    #se usa para que solo se cierre con la letra q y no se pueda cerrar por
    #accidente
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cuando todo esta listo, destruir ventadas abiertas y cerrar la camara
video_capture.release()
cv2.destroyAllWindows()




