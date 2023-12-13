import cv2 as cv

cap = cv.VideoCapture(1)

while True:
    ret,frame = cap.read()
    # resized = cv.resize(frame,(600,600))
    cv.imshow("FRAME",frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()    