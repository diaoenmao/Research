import cv2

path = 'input_video.mp4'
cap = cv2.VideoCapture(path)
print(cap.isOpened())   # True = read video successfully. False - fail to read video.

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_video.avi", fourcc, 20.0, (640, 360))
print(out.isOpened())  # True = write out video successfully. False - fail to write out video.
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret==True:
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()