from django.shortcuts import render

import pandas as pd
from PIL import Image, ImageFile
from PIL import Image
from scipy import ndimage
import numpy as np
import pydicom
from scipy import ndimage
from PIL import Image, ImageFile
import matplotlib.pylab as plt

#%matplotlib inline

from keras.models import load_model
#from tensorflow.python.keras.models import load_model
#import keras
#import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#import cv2


def prepare_dicom(dcm, width=None, level=None, norm=True):
    """
    Converts a DICOM object to a 16-bit Numpy array (in Hounsfield units)
    :param dcm: DICOM Object
    :return: Numpy array in int16
    """

    try:
        
        if dcm.BitsStored == 12 and dcm.PixelRepresentation == 0 and dcm.RescaleIntercept > -100:
            x = dcm.pixel_array + 1000
            px_mode = 4096
            x[x >= px_mode] = x[x >= px_mode] - px_mode
            dcm.PixelData = x.tobytes()
            dcm.RescaleIntercept = -1000

        pixels = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope + dcm.RescaleIntercept
    except ValueError as e:
        print("ValueError with", dcm.SOPInstanceUID, e)
        return np.zeros((512, 512))

    # Pad the image if it isn't square
    if pixels.shape[0] != pixels.shape[1]:
        (a, b) = pixels.shape
        if a > b:
            padding = ((0, 0), ((a - b) // 2, (a - b) // 2))
        else:
            padding = (((b - a) // 2, (b - a) // 2), (0, 0))
        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)
        
    if not width:
        width = dcm.WindowWidth
        if type(width) != pydicom.valuerep.DSfloat:
            width = width[0]
    if not level:
        level = dcm.WindowCenter
        if type(level) != pydicom.valuerep.DSfloat:
            level = level[0]
    lower = level - (width / 2)
    upper = level + (width / 2)
    img = np.clip(pixels, lower, upper)

    if norm:
        return (img - lower) / (upper - lower)
    else:
        return img

class CropHead(object):
    def __init__(self, offset=10):
        """
        Crops the head by labelling the objects in an image and keeping the second largest object (the largest object
        is the background). This method removes most of the headrest

        Originally made as a image transform for use with PyTorch, but too slow to run on the fly :(
        :param offset: Pixel offset to apply to the crop so that it isn't too tight
        """
        self.offset = offset

    def crop_extents(self, img):
        try:
            if type(img) != np.array:
                img_array = np.array(img)
            else:
                img_array = img

            labeled_blobs, number_of_blobs = ndimage.label(img_array)
            blob_sizes = np.bincount(labeled_blobs.flatten())
            head_blob = labeled_blobs == np.argmax(blob_sizes[1:]) + 1  # The number of the head blob
            head_blob = np.max(head_blob, axis=-1)

            mask = head_blob == 0
            rows = np.flatnonzero((~mask).sum(axis=1))
            cols = np.flatnonzero((~mask).sum(axis=0))

            x_min = max([rows.min() - self.offset, 0])
            x_max = min([rows.max() + self.offset + 1, img_array.shape[0]])
            y_min = max([cols.min() - self.offset, 0])
            y_max = min([cols.max() + self.offset + 1, img_array.shape[1]])

            return x_min, x_max, y_min, y_max
        except ValueError:
            return 0, 0, -1, -1

    def __call__(self, img):
        """
        Crops a CT image to so that as much black area is removed as possible
        :param img: PIL image
        :return: Cropped image
        """

        x_min, x_max, y_min, y_max = self.crop_extents(img)

        try:
            if type(img) != np.array:
                img_array = np.array(img)
            else:
                img_array = img

            return Image.fromarray(np.uint8(img_array[x_min:x_max, y_min:y_max]))
        except ValueError:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(offset={})'.format(self.offset)
#crop_head = CropHead()        

def dcm_to_png (dicom, crop, crop_head):
    r_dcm = pydicom.dcmread(dicom)
    g_dcm = pydicom.dcmread(dicom)
    b_dcm = pydicom.dcmread(dicom)
    r = prepare_dicom(r_dcm, width = 80, level = 40)
    g = prepare_dicom(g_dcm, width = 200, level = 80)
    b = prepare_dicom(b_dcm, width = 2000, level = 600)
    img = np.stack([r, g, b], -1)
    img = (img * 255).astype(np.uint8)
    im = Image.fromarray(img)

    if crop:
        x_min, x_max, y_min, y_max = crop_head.crop_extents(img > 0)
        img = img[x_min:x_max, y_min:y_max]

        if img.shape[0] == 0 or img.shape[1] == 0:
            img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

    #im = Image.fromarray(img.astype(np.uint8))

    return img

def modelprediction(Image1,model):   
    train_idg = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        shear_range=0.05,
        rotation_range=50, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,
        rescale=1./255)
    model = load_model(model)
    target_names = ['Any','Epidural','Intraparenchymal','Intraventricular', 'Subarachnoid','Subdural']
    predictions = []
    for i in range(7):
        im = train_idg.flow(Image1.reshape(1,224,224,3), batch_size=1, shuffle=False)
        im = next(im)
        pred = model.predict(im.reshape(1,224,224,3))
        predictions.append(pred)
    predictions = np.array(predictions)
    #print(predictions)
    predictions = np.where(predictions < 0.25, 0, 1)
    predictions = predictions.reshape(predictions.shape[0],predictions.shape[2])
    Result = pd.DataFrame(predictions.T,target_names).T
    Result = Result.mean(axis = 0)
    return pd.DataFrame(Result).T


def visualize(image):
    #plt.figure(figsize= (15,10))
    plt.imshow(image, interpolation='nearest')
    plt.tight_layout()
    plt.savefig('static/uploads/CTSCAN')    
    plt.show()
