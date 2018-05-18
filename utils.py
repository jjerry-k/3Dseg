import glob
import cv2
import numpy as np
import pydicom as di
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def plot_3d(_3d):
    '''
    _3d : [depth, height, width] = [z, y, x]
    '''
    
    in_shape = _3d.shape
    
    _x = []
    _y = []
    _z = []
    
    # k : depth, j : height, i : width
    for kdx in range(in_shape[0]):
        for jdx in range(in_shape[1]):
            for idx in range(in_shape[2]):
                if _3d[kdx,jdx,idx] < 1. : continue
                else:
                    _x.append(idx)
                    _y.append(jdx)
                    _z.append(kdx)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(_x, _y, _z, c="k", marker="o")
    
    ax.set_xlim(-2,in_shape[2]+2)
    ax.set_ylim(-2,in_shape[1]+2)
    ax.set_zlim(-2,in_shape[0]+2)
    
    plt.show()


def read_dicom_image(dcm_paths):
    img = []
    for idx, path in enumerate(dcm_paths):
        tmp = di.read_file(path)
        tmp_img = tmp.pixel_array
        img.append(tmp_img)
    img = np.array(img)
    return(img)

def read_csv(csv_path):
    coor_csv = []
    for i in csv_path:
        csv = pd.read_csv(i)
        for idx, val in csv.iterrows():
            coor_csv.append(np.array([[val['X']],[val['Y']],[val['Z']],[1.]]))

    coor_csv = np.reshape(np.array(coor_csv), [-1,4])
    coor_csv = np.transpose(coor_csv, [1,0])
    
    return coor_csv



def make_affine_matrix(dcm_paths):
    
    '''
    input 
        - dcm_paths : DICOM files paths
        
    output
        - 3D Affine Formulae
    '''
    
    tmp=di.read_file(dcm_paths[0])
    '''
    A1 = np.zeros([4,4])
    A1[:3, 0] = np.array(tmp.ImageOrientationPatient)[3:]
    A1[:3, 1] = np.array(tmp.ImageOrientationPatient)[:3]
    A1[:3, 3] = np.array(tmp.ImagePositionPatient)
    A1[-1,-1] = 1
    '''
    T1 = np.array(tmp.ImagePositionPatient)
    
    tmp = di.read_file(dcm_paths[-1])
    '''
    AN = np.zeros([4,4])
    AN[:3, 0] = np.array(tmp.ImageOrientationPatient)[3:]
    AN[:3, 1] = np.array(tmp.ImageOrientationPatient)[:3]
    AN[:3, 3] = np.array(tmp.ImagePositionPatient)
    AN[-1,-1] = 1
    '''
    TN = np.array(tmp.ImagePositionPatient)
    
    spacing = tmp.PixelSpacing
    
    AM = np.zeros([4,4])
    AM[-1,-1] = 1
    AM[:3, 0] = np.array(tmp.ImageOrientationPatient)[3:]*spacing[0]
    AM[:3, 1] = np.array(tmp.ImageOrientationPatient)[:3]*spacing[1]
    AM[:3, 2] = np.reshape((T1-TN)/(1-len(dcm_paths)), (-1))[:3]
    AM[:3, 3] = np.reshape(T1[:3], -1)
    
    return AM

def make_data(to_path, num, data, use_scale = True):
    
    dcm_paths = glob.glob(to_path+'%s/%s/MR*.dcm'%(num, data))
    
    dcm_paths = extract_dicom_list(dcm_paths)
    
 
    '''
    if num != '01':
        ind = ind = indices[int(data)]
        dcm_paths = dcm_paths[ind[0]:ind[1]]
    '''
    csv_path = glob.glob(to_path+'%s/%s/*.csv'%(num, data))
           
    img = read_dicom_image(dcm_paths)
    

    affine = make_affine_matrix(dcm_paths)
    inv_affine = np.linalg.inv(affine)
        
    coor = read_csv(csv_path)
        
    ori_coor=np.uint(np.matmul(inv_affine, coor))
    # Make label data

    ground = np.zeros(img.shape)
    for i in range(len(ori_coor[0])):
        ground[ori_coor[2,i], ori_coor[0,i], ori_coor[1,i]]=1
    
    if use_scale == False:
        return img, ground, ori_coor
    
    else:
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        return img, ground, ori_coor

    
def resizing_data(data, size):
    out = np.zeros([data.shape[0], size, size])
    for i in range(data.shape[0]):
        out[i] = cv2.resize(data[i], (size,size))
        
    return out
        
def extract_dicom_list(dicom_paths):
    
    path_list = []
    
    dicom_paths.sort()
        
    if 'RTST' in dicom_paths[-1]:
        dicom_paths.pop()
    cnt = 0
    for dicom in dicom_paths:
        dcm = di.read_file(dicom)
        if 'MPR' not in dcm.SeriesDescription.upper(): continue
        elif 'HOSPITAL' in dcm.SeriesDescription.upper() : continue
        elif dcm.SliceThickness != 1 : continue
        else:
            path_list.append(dicom)
    return path_list

def extend(x):

    '''
    Extend tensor X
    b, d, h, w, c => b, 2d, 2h, 2w, c

    '''

    b, d, h, w, c = x.get_shape().as_list()



    out = tf.transpose(x, [0,4,1,2,3])       # (b, d, h, w, c) - > (b, c, d, h, w)
    out = tf.reshape(out, [-1,1])          # (b, c, d, h, w) - > (b*c*d*h*w, 1)
    out = tf.matmul(out, tf.ones([1,2]))      # (b*c*d*h*w, 1) - > (b*c*d*h*w, 2)


    out = tf.reshape(out, [-1, c, d, h, 2*w])  # (b*c*d*h*w, 2) - > (b, c, h, 2*w)
    out = tf.transpose(out, [0,1,2,4,3])     # (b, c, d, h, 2*w) - > (b, c, d, 2*w, h)
    out = tf.reshape(out, [-1,1])          # (b, c, d, 2*w, h) - > (b*c*d*2w*h, 1)
    out = tf.matmul(out, tf.ones([1,2]))      # (b*c*d*2w*h, 1) - > (b*c*d*2w*h, 2)


    out = tf.reshape(out, [-1, c, d, 2*h, 2*w]) # (b*c*d*2w*h, 2) - > (b, c, d, 2w, 2h)
    out = tf.transpose(out, [0, 1, 3, 4, 2])  # (b, c, d, 2w, 2h) - > (b, c, 2h, 2w, d)
    out = tf.reshape(out, [-1,1])          # (b, c, 2*w, 2h, d) - > (b*c*2w*2h*d, 1)
    out = tf.matmul(out, tf.ones([1,2]))      # (b*c*2w*2h*d, 1) - > (b*c*2w*2h*d, 2)

    out = tf.reshape(out, [-1, c, 2*h, 2*w, 2*d]) # (b*c*2w*2h*d, 2) -> (b, c, 2w, 2h, 2d)
    out = tf.transpose(out, [0, 4, 2, 3, 1])

    return out



def extract_loc(x, y):

    '''
    x : input data at pooling layer
    y : extending pooling data
    x'shape == y'shape

    '''

    out = tf.equal(x, y)                  # tf.equal([[1,1],[3,4]], [[4,4],[4,4]]) = [[False, False],[False, True]]
    out = tf.cast(out, dtype=tf.float32)  # tf.cast([[False, False],[False, True]], dtype = tf.float32) = [[0.,0.],[0.,1.]]


    return out

def unpool3d(x, y):
    _x = extend(x)
    out = extract_loc(_x, y)
    return out

def init_w(name, shape):
    '''
    shape : [filter_h, filter_w, input_channel, ouput_channel]
    '''
    w = tf.get_variable(name, shape=shape,
                          initializer=tf.contrib.layers.xavier_initializer())
    return w


def max_pool_3d(_input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="SAME"):
    pool = tf.nn.max_pool3d(_input,  ksize=ksize, strides=strides, padding="SAME")
    return pool


def conv3d(_input, weight, strides = [1,1,1,1,1], padding = "SAME"):
    conv = tf.nn.conv3d(_input, weight, strides=strides, padding = padding)
    return conv



def deconv3d(_input, weight, _output_shape = None, strides = [1, 1, 1, 1, 1], padding = "SAME"):
    _input_shape = _input.shape.as_list()
    weight_shape = weight.shape.as_list()

    if _output_shape == None:
        _output_shape = [tf.shape(_input)[0], _input_shape[1], _input_shape[2], _input_shape[3], weight_shape[3]]
    add_zero = tf.zeros([1, _output_shape[1], _output_shape[2], _output_shape[3], _output_shape[4]])
    deconv = tf.nn.conv3d_transpose(_input, weight, output_shape=_output_shape, strides=strides, padding=padding)
    return (deconv + add_zero)

def max_pool_2d(_input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"):
    pool = tf.nn.max_pool(_input,  ksize=ksize, strides=strides, padding="SAME")
    return pool


def conv2d(_input, weight, strides = [1,1,1,1], padding = "SAME"):
    conv = tf.nn.conv2d(_input, weight, strides=strides, padding = padding)
    return conv



def deconv2d(_input, weight, _output_shape = None, strides = [1, 1, 1, 1], padding = "SAME"):
    _input_shape = _input.shape.as_list()
    weight_shape = weight.shape.as_list()

    if _output_shape == None:
        _output_shape = [tf.shape(_input)[0], _input_shape[1], _input_shape[2], weight_shape[3]]
    add_zero = tf.zeros([1, _output_shape[1], _output_shape[2], _output_shape[3],])
    deconv = tf.nn.conv2d_transpose(_input, weight, output_shape=_output_shape, strides=strides, padding=padding)
    return (deconv + add_zero)


def batch_norm(_input, center=True, scale=True, decay=0.8, is_training=True):
    norm = tf.contrib.layers.batch_norm(_input, center=center, scale = scale,
                                        decay = decay, is_training=is_training)
    return norm

