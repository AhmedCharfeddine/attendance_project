import cv2
import numpy as np
import face_recognition as fr
import os


print("importation des images des étudiants")
path = "images_des_étudiants"
images = [] # liste pour les images (des arrays)
liste_inscrits = [] # contient les noms des inscrits



L = os.listdir(path) # les éléments de L sont les noms des images dans images (avec l'extension .png)
for fileName in L:
    images += [cv2.imread("{}/{}".format(path, fileName))]
    liste_inscrits += [fileName.split('.')[0]]
#liste_inscrits: contient les noms des étudiants



print("codage...")
liste_présence_encoded = []
for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        liste_présence_encoded += [fr.face_encodings(img)[0]]
print('codage terminé')


attending = set() #liste des étudiants présents
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)     # réduit la taille pour accélérer la procédure
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # on prendra le cas si plusieurs visages sont présents devant la caméra

    facesInFrame =fr.face_locations(imgS)  # liste contenant les rectangeles de tous les visages présents devant la caméra
    frameEncoded = fr.face_encodings(imgS, facesInFrame)  # les visages présents encodés dans la liste facesInFrame

    for faceEncoded, faceLoc in zip(frameEncoded, facesInFrame):
        facedistances = fr.face_distance(liste_présence_encoded, faceEncoded)
        # liste contenant des valeurs représentant les distances entre les visages du visage au webcam et ceux des inscrits
        # plus la valeur est petite plus la corrélation est importante. on prendra 0.6 comme valeur de tolérance (valeur par défaut)
        if (np.min(facedistances)>0.6):
            print("un visage n'a pas été reconnu")
        else: # visage reconnu
            matching_face_index = np.argmin(facedistances)
            #index de l'image ayant la plus petite valeur
            nom = liste_inscrits[matching_face_index]
            if not nom in attending: # si l'étudiant devant la caméra est reconnu mais non déjà marqué présent:
                attending |= {nom} #le marquer présent
                print("l'étudiant {} est marqué présent".format(nom))


    if len(attending) == len(liste_inscrits):
        print("tous les étudiants sont présents")
        break

cap.release() #fermer la caméra
