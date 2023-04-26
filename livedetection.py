import cv2

image = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    dummy,frame = image.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    myface = classifier.detectMultiScale(grey,1.1,5)
    print(myface)
    print(len(myface))
    for x,y,w,h in myface:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,169),2)
        rvoi = frame[y:y+h,x:x+w]
    cv2.imshow('faceid', frame)
    if cv2.waitKey(25) == 32:
        break

image.release()
cv2.destroyAllWindows()

