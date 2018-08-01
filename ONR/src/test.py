import numpy as np
import cv2
import base64

imagefilename = 'lena.tif'
img=cv2.imread(imagefilename, 1)
ret,img_encode = cv2.imencode('.jpg', img)
binary_img_encode = img_encode.tostring()
with open('lena.jpg', 'wb') as f:
    f.write(binary_img_encode)
with open('lena.jpg', 'rb') as f:
    binary_img_encode_r = f.read()
img_encode_r = np.frombuffer(binary_img_encode_r, dtype=np.uint8);    
img_decode = cv2.imdecode(img_encode_r, cv2.IMREAD_COLOR)
print(img_encode_r)
assert np.array_equal(img_encode.reshape(-1),img_encode_r)

# path = 'input_video.mp4'
# cap = cv2.VideoCapture(path)
# print(cap.isOpened())   # True = read video successfully. False - fail to read video.
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("output_video.avi", fourcc, 20.0, (640, 360))
# print(out.isOpened())  # True = write out video successfully. False - fail to write out video.
# while(cap.isOpened()):
    # ret,frame = cap.read()
    # if ret==True:
        # out.write(frame)
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    # else:
        # break
# cap.release()
# out.release()
# cv2.destroyAllWindows()


