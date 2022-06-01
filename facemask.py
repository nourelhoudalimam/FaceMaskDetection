
import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as k
from tensorflow.keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import datetime
import matplotlib.pyplot as plt

# UNCOMMENT THE FOLLOWING CODE TO TRAIN THE CNN FROM SCRATCH

# BUILDING MODEL TO CLASSIFY BETWEEN MASK AND NO MASK

model=Sequential() #creation modele basique de CNN
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3))) #commencer notre modele par une couche de convolution,l'application d'un filtre pour extraire les caracteristiques d'une image
#on va utiliser 32 filtres
#activation='relu' pour rendre les valeurs negatifs dans la matrice d'image nulles
model.add(MaxPooling2D() ) #selectionner l'intensite plus grande parmi une selection de pixels
model.add(Conv2D(32,(3,3),activation='relu')) #ajouter une autre couche de convolution en remplacant toute resultat negatif en 0
model.add(MaxPooling2D() ) 
model.add(Conv2D(32,(3,3),activation='relu')) #ajouter une autre couche de convolution en remplacant toute resultat negatif en 0
model.add(MaxPooling2D() )#réduit la dimensionnalité une autre fois des images en réduisant le nombre de pixels dans la sortie de la couche convolutive précédente.
model.add(Flatten()) #transformer image en une vecteur
model.add(Dense(100,activation='relu')) #La couche dense est une couche de réseau de neurones profondément connectée, ce qui signifie que chaque neurone de la couche dense reçoit une entrée de tous les neurones de sa couche précédente. 
#en arrière-plan, la couche dense effectue une multiplication matrice-vecteur. Les valeurs utilisées dans la matrice sont en fait des paramètres qui peuvent être formés et mis à jour à l'aide de la rétropropagation.
#la sortie generee par la couche dense est un vecteur de dimension « m ». Ainsi, la couche dense est essentiellement utilisée pour modifier les dimensions du vecteur. Les couches denses appliquent également des opérations telles que la rotation, la mise à l'échelle, la translation sur le vecteur.
#100 represents the output size of the layer.
model.add(Dense(1,activation='sigmoid')) #ajouter une autre couche Dense 
#1 represents the output size of the layer.
#Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)). 
#Applique la fonction d'activation sigmoïde. Pour les petites valeurs (<-5), sigmoïde renvoie une valeur proche de zéro, et pour les grandes valeurs (>5), le résultat de la fonction se rapproche de 1.
#Sigmoïde équivaut à un Softmax à 2 éléments, où le deuxième élément est supposé être nul. La fonction sigmoïde renvoie toujours une valeur comprise entre 0 et 1.
#La fonction d'activation sert avant tout à modifier de manière non-linéaire les données. Cette non-linéarité permet de modifier spatialement leur représentation. Dit simplement, la fonction d'activation permet de changer notre manière de voir une donnée.

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #compiler le modele
#You need a compiled model to train (because training uses the loss function and the optimizer).
#optimizer='adam' :L'algorithme d'optimisation Adam est utilisé pour la formation de modèles DEEP LEARNING. Il s'agit d'une extension de la descente de gradient stochastique. Dans cet algorithme d'optimisation, les moyennes courantes des gradients et des seconds moments des gradients sont utilisées. Il est utilisé pour calculer les taux d'apprentissage adaptatifs pour chaque paramètre
#loss='binary_crossentropy' : The binary_crossentropy function computes the cross-entropy loss between true labels and predicted labels. categorical_crossentropy: Used as a loss function for multi-class classification model where there are two or more output labels.
#metrics=['accuracy'] : evaluer les performances de votre modele .  Calcule la fréquence à laquelle les prédictions correspondent aux libellés. Cette métrique crée deux variables locales, total et count qui sont utilisées pour calculer la fréquence à laquelle y_pred correspond à y_true . Cette fréquence est finalement renvoyée sous forme de précision binaire : une opération idempotente qui divise simplement le total par le nombre .
model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#La classe ImageDataGenerator vous permet de faire pivoter de manière aléatoire des images sur n'importe quel degré entre 0 et 360 en fournissant une valeur entière dans l'argument rotation_range.
#Rescale 1./255 is to transform every pixel value from range [0,255] -> [0,1]. And the benefits are: Treat all images in the same manner: some images are high pixel range, some are low pixel range.
#shear_range spécifie l'angle de l'inclinaison en degrés.
#zoom_range signifie zoom avant et zoom arriere de 20%
#horizontal_flip=True retourne essentiellement les lignes et les colonnes horizontalement. 
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=16 ,
        class_mode='binary')
#flow_from_directory : Cette méthode est utile lorsque les images sont triées et placées dans leurs dossiers de classe/étiquette respectifs. Cette méthode identifiera automatiquement les classes à partir du nom du dossier.
#target_size: est la taille de vos images d'entrée, chaque image sera redimensionnée à cette taille.
#batch_size :Nombre d'images à générer à partir du générateur par lot.
#class_mode='binary': car on a 2 classes à prevoir (test et train)
test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')
model_saved=model.fit_generator(
        training_set,
        epochs=20,callbacks=[callback],
        validation_data=test_set)
#fit_generator or fit () are two separate deep learning libraries which can be used to train our machine learning and deep learning models. Both these functions can do the same task, but when to use which function is the main question
#training_set is the model to train / epochs est le nombre d'époques pour lesquelles nous voulons former notre modèle. /
#validation_data utilisée pour évaluer la perte et les métriques pour n'importe quel modèle après la fin de n'importe quelle époque.
model.save('modelFaceMask.h5',model_saved) #enregistrer le model créé

#To test for individual images

mymodel=load_model('modelFaceMask.h5') #loading the model created
#test_image=image.load_img('C:/Users/Karan/Desktop/ML Datasets/Face Mask Detection/Dataset/test/without_mask/30.jpg',target_size=(150,150,3))
test_image=image.load_img(r'/home/nour/Downloads/PFA/pfa/test/without_mask/0.jpg',
                          target_size=(150,150,3)) #loading image from the dataset with target size (150,150,3)
test_image #ouvrir cette image
test_image=image.img_to_array(test_image) #convertir l'image de dataset en numpy array
test_image=np.expand_dims(test_image,axis=0) #Développer la forme d'un numpy array.
#Insérez un nouvel axe (axis=0 colonne) qui apparaîtra à la position de l'axe dans la forme de tableau développé.
mymodel.predict(test_image)[0][0] #passes the input vector through the model and returns the output tensor for each datapoint. 


# IMPLEMENTING LIVE DETECTION OF FACE MASK

mymodel=load_model('modelFaceMask.h5') #loading model created

cap=cv2.VideoCapture(0) #capturer un video à l'aide de VideoCapture de opencv
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Il s'agit d'une approche basée sur l'apprentissage automatique dans laquelle une fonction en cascade est formée à partir d'un grand nombre d'images positives et négatives. Il est ensuite utilisé pour détecter des objets dans d'autres images.

while cap.isOpened(): #tant que camera est ouvert
    _,img=cap.read() #le cadre lit correctement
    face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)#détecte des objets de différentes tailles dans l'image       d'entrée. Les objets détectés sont renvoyés sous la forme d'une liste de rectangles. Matrice de type CV_8U contenant une image où des objets sont détectés.
    for(x,y,w,h) in face: #(x,y,w,h) sont des dimensions x:longueur/y:largeur/h:hauteur/w:poids
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('tempmask.jpg',face_img)
        test_image=image.load_img('tempmask.jpg',target_size=(150,150,3))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        pred=mymodel.predict(test_image)[0][0]
        training_set.class_indices
        if pred==1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
          
    cv2.imshow('img',img) #est utilisé pour afficher img dans une fenêtre. La fenêtre s'adapte automatiquement à la taille de l'image.
    
    if cv2.waitKey(1)==ord('q'): #si on tape à q on va fermer le camera et finir l'execution
        break


plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(model_saved.history['loss'], label='Training Loss')
plt.plot(model_saved.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(model_saved.history['accuracy'], label='Training Accuracy')
plt.plot(model_saved.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()    
cap.release() #liberer ressources materielles et logicielles
cv2.destroyAllWindows() #openCV va détruire toutes les fenêtres que nous avons créées.


