import cv2
import numpy as np
import face_recognition as fr

imgperson = fr.load_image_file("img/jason1.jpg")
imgperson = cv2.cvtColor(imgperson, cv2.COLOR_BGR2RGB)
imgperson2 = fr.load_image_file("img/jason2.jpg")
imgperson2 = cv2.cvtColor(imgperson2, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgperson)[0]
encodeperson = fr.face_encodings(imgperson)[0]
cv2.rectangle(imgperson, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255),2)


faceLoc2 = fr.face_locations(imgperson2)[0]
encodeperson2 = fr.face_encodings(imgperson2)[0]
cv2.rectangle(imgperson2, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255,0,255),2)

results = fr.compare_faces([encodeperson], encodeperson2)
print(results)
cv2.putText(imgperson2, "{}".format(results), (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


print(faceLoc)
cv2.imshow("jason statham", imgperson)
cv2.imshow("jason statham 2", imgperson2)
cv2.waitKey(0)

