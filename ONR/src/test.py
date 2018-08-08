import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from util import *
from torchvision.utils import make_grid
from util import *
from data import *

# imagefilename = './data/sample/lena.tif'
# img=cv2.imread(imagefilename, 1)
# ret,img_encode = cv2.imencode('.jpg', img)
# binary_img_encode = img_encode.tostring()
# with open('./data/sample/lena.jpg', 'wb') as f:
    # f.write(binary_img_encode)
# with open('./sample/lena.jpg', 'rb') as f:
    # binary_img_encode_r = f.read()
# img_encode_r = np.frombuffer(binary_img_encode_r, dtype=np.uint8);    
# img_decode = cv2.imdecode(img_encode_r, cv2.IMREAD_COLOR)
# print(img_encode_r)
# assert np.array_equal(img_encode.reshape(-1),img_encode_r)

# path = './data/sample/input_video.mp4'
# cap = cv2.VideoCapture(path)
# print(cap.isOpened())   # True = read video successfully. False - fail to read video.
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("./sample/output_video.avi", fourcc, 20.0, (640, 360))
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




# N = 10
# C = 3
# H = 1023
# W = 800
# x = torch.randn(N,C,H,W)
# print(x.shape)
# size = (128,190)
# patches_fold_H = x.unfold(2, size[0], size[0])
# print(patches_fold_H.shape)
# if(H % size[0] != 0):
    # patches_fold_H = torch.cat((patches_fold_H,x[:,:,-size[0]:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
# print(patches_fold_H.shape)
# patches_fold_HW = patches_fold_H.unfold(3, size[1], size[1])
# if(W % size[1] != 0):
    # patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-size[1]:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
# print(patches_fold_HW.shape)
# patches = patches_fold_HW.permute(0,2,3,1,4,5).reshape(-1,C,size[0],size[1])
# print(patches.shape)


# size = (128,128)
# #size = (500,800)
# imagefilename = 'data/sample/mountain.png'
# img_np = cv2.cvtColor(cv2.imread(imagefilename, 1), cv2.COLOR_BGR2RGB)
# img = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0)
# print(img.shape)
# patches = extract_patches_2D(img,size)
# print(patches.shape)
# nrow = int(np.ceil(float(img.size(3))/size[1]))
# show_patches = make_grid(patches,nrow=nrow).permute(1,2,0).numpy()
# plt.imshow(show_patches)
# plt.show()

# size = (128,128)
# imagefilename = 'data/sample/mountain.png'
# BGR_img = cv2.imread(imagefilename, 1)
# YCC_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2YCR_CB)
# my_YCC_img = RGB_to_YCbCr(torch.from_numpy(np.float32(BGR_img)).permute(2,0,1).unsqueeze(0)).squeeze().permute(1,2,0).numpy()
# assert np.array_equal(YCC_img,my_YCC_img)
# my_RGB_img = YCbCr_to_RGB(RGB_to_YCbCr(torch.from_numpy(np.float32(BGR_img)).permute(2,0,1).unsqueeze(0))).squeeze().permute(1,2,0).numpy()
# my_BGR_img = cv2.cvtColor(BGR_img, cv2.COLOR_RGB2BGR)
# assert np.array_equal(BGR_img,my_BGR_img)


# path = './data/ImageNet/train'
# unzip(path,'tar')

# path = './data/sample/lena.jpg'
# img = Image.open(path)
# transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.Lambda(lambda x: RGB_to_YCbCr(x))])
# transformed_img = transform(img)
# copy_image = np.array(transformed_img.copy()) # Make a copy
# copy_image[:,:,0] = 0
# copy_image[:,:,1] = 0
# plt.imshow(copy_image)
# plt.show()

# path = './data/sample/lena.jpg'
# img = Image.open(path)
# transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.Lambda(lambda x: RGB_to_YCbCr(x)),
                # transforms.ToTensor()])
# transformed_img = transform(img)
# print(transformed_img.size())
