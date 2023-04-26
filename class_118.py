import cv2

image = cv2.imread('4f.jpg')
grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
myface = classifier.detectMultiScale(grey,1.1,5)
print(myface)
print(len(myface))
for x,y,w,h in myface:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,169),2)
    rvoi = image[y:y+h,x:x+w]
    cv2.imwrite('myface.jpg',rvoi)
cv2.imshow('faceid', image)
cv2.waitKey(0)
