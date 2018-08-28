import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from util import *



def main():
    jpeg_rate_distortion = load('./output/result/0_MNIST_jpeg.pkl')['rate_distortion']
    pure_rate_distortion = load('./output/result/0_MNIST_RCAE_pure.pkl')['rate_distortion']
    handtuned_rate_distortion = load('./output/result/0_MNIST_RCAE_handtuned.pkl')['rate_distortion']
    handtuned_acc = load('./output/result/0_MNIST_RCAE_handtuned.pkl')['acc']
    print('handtuned_acc: {}'.format(handtuned_acc))
    plt.figure()
    plt.plot(jpeg_rate_distortion[0,:],jpeg_rate_distortion[1,:],color='r',label='jpeg')
    plt.plot(pure_rate_distortion[0,:],pure_rate_distortion[1,:],color='g',label='pure')
    plt.plot(handtuned_rate_distortion[0,:],handtuned_rate_distortion[1,:],color='b',label='handtuned')
    plt.xlabel('bpp')
    plt.ylabel('psnr(db)')
    plt.grid()
    plt.legend()
    plt.show()




























if __name__ == "__main__":
    main()
