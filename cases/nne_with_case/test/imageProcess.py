import cv2
import numpy as np
from ctypes import *
import pytest
import math
import random
import struct
import sys


mChannels = 3
mMean = [0.1, 0.2, 0.3]
mStd = [1., 1., 1.]
input_height = 1080
input_width = 1920

split = True


def resize_with_pad(img, mWidth, mHeight):
    print(mWidth, mHeight)
    scale = min(mWidth / input_width, mHeight / input_height)
    scale_size = (int(input_width * scale), int(input_height * scale))
    resized = cv2.resize(img, scale_size)
    out = np.zeros((mHeight, mWidth), dtype=np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    out[(mHeight - scale_size[1]) // 2 : (mHeight - scale_size[1]) // 2 + scale_size[1],
        (mWidth - scale_size[0]) // 2 : (mWidth - scale_size[0]) // 2 + scale_size[0],:] = resized
    #cv2.imwrite("1920x1080_3c_resize.jpg", out)
    return out


@pytest.mark.run_rgb_resize_norm
def test_rgb_resize_norm_cv():
    mWidth = int(sys.argv[2])
    mHeight = int(sys.argv[3])
    img = cv2.imread("../res/1920x1080_3c.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fp = open("input_data.txt", 'wb')
    fp.write(img)
    fp.close()
    cropped = resize_with_pad(img, mWidth, mHeight)
    img_float = cropped * (1/255)

    if split == True:
        input_channels = cv2.split(img_float)
        normed_channel = np.zeros((mChannels, mHeight * mWidth), dtype=np.float)
        c_idx = 0
        for input_channel in input_channels:
            normed_channel[c_idx] = (input_channel.reshape(mHeight * mWidth) - mMean[c_idx]) / mStd[c_idx]
            c_idx += 1
        data = normed_channel.reshape(mChannels * mHeight * mWidth)
    else:
        data = cropped.reshape(mChannels * mHeight * mWidth) * 0.003921
    fp = open("out_data.txt", 'wb')
    fp.write(data.astype(np.float32))
    fp.close()



@pytest.mark.run_nv122rgb_resize_norm
def test_nv122rgb_resize_norm_cv(IsRGB):
    mWidth = int(sys.argv[2])
    mHeight = int(sys.argv[3])
    fp = open("../res/1920x1080_nv12.yuv", "rb")
    img = fp.read()
    data = struct.unpack('B' * len(img), img)
    data = np.uint8(data)
    data = np.asarray(data)
    data = data.reshape((1080 * 3 // 2, 1920))
    if IsRGB == True:
        img = cv2.cvtColor(data, cv2.COLOR_YUV2RGB_NV12)
    else:
        img = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_NV12)

    cropped = resize_with_pad(img, mWidth, mHeight)
    #img_float = cropped * (1/255)
    img_float = cropped * 1.0
    if split == True:
        input_channels = cv2.split(img_float)
        normed_channel = np.zeros((mChannels, mHeight * mWidth), dtype=np.float)
        c_idx = 0
        for input_channel in input_channels:
            normed_channel[c_idx] = (input_channel.reshape(mHeight * mWidth) - mMean[c_idx]) / mStd[c_idx]
            c_idx += 1
        data = normed_channel.reshape(mChannels * mHeight * mWidth)
    else:
        data = cropped.reshape(mChannels * mHeight * mWidth) * 0.003921
    fp = open("out_data_nv12.txt", 'wb')
    fp.write(data.astype(np.float32))
    fp.close()
    

@pytest.mark.run_yu122rgb_resize_norm
def test_yu122rgb_resize_norm_cv(IsRGB):
    mWidth = int(sys.argv[2])
    mHeight = int(sys.argv[3])
    fp = open("../res/1920x1080_yu12.yuv", "rb")
    img = fp.read()
    data = struct.unpack('B' * len(img), img)
    data = np.uint8(data)
    data = np.asarray(data)
    data = data.reshape((1080 * 3 // 2, 1920))
    if IsRGB == True:
        img = cv2.cvtColor(data, cv2.COLOR_YUV2RGB_I420)
    else:
        img = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_I420)

    cropped = resize_with_pad(img, mWidth, mHeight)
    #img_float = cropped * (1/255)
    img_float = cropped * 1.0
    if split == True:
        input_channels = cv2.split(img_float)
        normed_channel = np.zeros((mChannels, mHeight * mWidth), dtype=np.float)
        c_idx = 0
        for input_channel in input_channels:
            normed_channel[c_idx] = (input_channel.reshape(mHeight * mWidth) - mMean[c_idx]) / mStd[c_idx]
            c_idx += 1
        data = normed_channel.reshape(mChannels * mHeight * mWidth)
    else:
        data = cropped.reshape(mChannels * mHeight * mWidth) * 0.003921
    fp = open("out_data_yu12.txt", 'wb')
    fp.write(data.astype(np.float32))
    fp.close()


if __name__ == "__main__":
    case_name = sys.argv[1]
    if case_name == "rgb_resize_norm":
        test_rgb_resize_norm_cv()
    elif case_name == "nv122rgb_resize_norm":
        test_nv122rgb_resize_norm_cv(True)
    elif case_name == "yu122rgb_resize_norm":
        test_yu122rgb_resize_norm_cv(True)
    elif case_name == "nv122bgr_resize_norm":
        test_nv122rgb_resize_norm_cv(False)
    elif case_name == "yu122bgr_resize_norm":
        test_yu122rgb_resize_norm_cv(False)
    
