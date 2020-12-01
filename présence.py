import cv2
import numpy as np
import face_recognition as fr
import os

def encodelist(images): #retourne une liste de vecteurs des images encodées fournies
    res = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res += [fr.face_encodings(img)[0]]
    return res

print("getting liste_présence")
path = "img"
images = [] #liste pour les images (des arrays)
liste_inscrits = [] #contient les noms des inscrits
L = os.listdir(path)
# les éléments de L sont les noms des images dans liste_présence

for fileName in L:
    images += [cv2.imread("{}/{}".format(path, fileName))]
    liste_inscrits += [fileName.split('.')[0]]

print("encoding")
liste_présence_encoded = encodelist(images)
print('encoding done')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)     #réduit la taille pour accélérer la procédure
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    #on prend le cas si plusieurs visages sont présents devant la caméra

    facesInFrame =fr.face_locations(imgS)       #liste contenant les rectangeles de tous les visages présents devant la caméra
    frameEncoded = fr.face_encodings(imgS, facesInFrame)     #les visages présents encodés dans la liste facesInFrame


    print("j'ai trouvé {} visages".format(len(facesInFrame)))

    for faceEncoded, faceLoc in zip(frameEncoded, facesInFrame):
        print("type(frameEncoded) is {}".format(type(frameEncoded)))
        matches = fr.compare_faces(np.array(liste_présence_encoded), np.array(frameEncoded))
        #liste contenant des True ou False si le visage du caméra ressemble à un élément des inscrits
        if True not in matches:
            print("Ce visage n'a pas été reconnu")
        else:
            facedistances = fr.face_distance(imgS, facesInFrame)
            #liste ayant des valeurs décrivant la corrélation du visage présenté avec les visages du liste
            #plus la valeur est petite plus la corrélation est importante
            matching_face_index = np.argmin(facedistances)
            #index de l'image ayant la plus petite valeur
            nom = liste_inscrits[matching_face_index]
            print(nom)
