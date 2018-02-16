import cv2
import time
import os


video_capture = cv2.VideoCapture(0)#Se inicia la captura con la camara

nom=raw_input('ingrese su nombre: ')#se pide el nombre para crear la carpeta

#se crea un directorio para guardar el cnjunto de imagenes
newpath = 'C:/Python27/imagen/dataSet'
newpath=newpath +'/'+ nom

# se realiza la comparacion de si existe el directorio, y si no, lo crea
if not os.path.exists(newpath):
    os.makedirs(newpath)

#se lee la camara y se captura un frame  
ret, frame = video_capture.read()
#se realiza un flip para que quede como espejo
frame=cv2.flip(frame,1,0)

# se muestra la imagen tomada para que la persona se ubique
cv2.imshow('Frame', frame)
print 'Press any key to begin'
cv2.waitKey(0)
cv2.destroyAllWindows()

#se realiza un ciclo for de 200 iteraciones para tomar 200 fotos
for i in range(200):
    ret, frame = video_capture.read()#se lee la camara y se captura un frame
    frame=cv2.flip(frame,1,0)#se realiza un flip para que quede como espejo
    print i #se imprime el numero de iteraciones para saber cuantas fotos lleva

    #se guarda la imagen en el directorio seleccionado, y se cambia el nombre
    #a medida que se repite el ciclo for
    cv2.imwrite(newpath+"/"+nom+'_'+ str(i) + ".jpg", frame)
    #se realiza un delay de 0.1 seg entre captura y captura
    time.sleep(0.1)
    

print '[DONE]' # se imprime DONE para que el usuario sepa que ha terminado
video_capture.release()# se libera la camara

