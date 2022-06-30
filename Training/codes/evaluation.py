import warnings
import numpy as np
import os
import sys
import glob
from random import randint
from sklearn import metrics
import re
import os
import numpy as np
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import Progbar
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from os.path import exists
import cv2
import numpy as np
from torchsummary import summary
import scipy.ndimage as ndi
from torchvision import transforms
import json
from torch.utils.data import Dataset, DataLoader ,random_split

test_dir = '/home/rtr/Powerline_Tracking/datasets/testing'
print(test_dir)
experiment_rootdir = '/home/rtr/Powerline_Tracking/models/model1'
PATH = experiment_rootdir + '/' +'Powerline_Tracking_epoch-10-0.0015631477581337094.pth'

print('PATH:', PATH)
IN_CHANNELS = 1
IMG_WIDTH = 200
IMG_HEIGHT = 200

img_width, img_height = 320,240
crop_img_width, crop_img_height = 200,200

# target_size=(320,240)
target_size=(200,200)

crop_size = (200,200)
color_mode = 'grayscale'
batch_size = 16

class DroneDirectoryIterator():
    global target_size , crop_size, color_mode, batch_size
    def __init__(self, directory, target_size=target_size, 
                    crop_size = crop_size, color_mode=color_mode, batch_size = 32, 
                    shuffle=False, seed=None, follow_links=False):
        self.directory = directory
        self.shuffle = shuffle
        self.seed = seed
        self.target_size = tuple(target_size)
        self.crop_size = tuple(crop_size)
        self.follow_links = follow_links

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = self.crop_size + (3,)
        else: #grayscale
            self.image_shape = self.crop_size + (1,)

        self.samples = 0

        experiments = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                experiments.append(subdir)

        self.num_experiments = len(experiments)

        self.formats = {'png'}
        
        # Idea = associate each filename with a corresponding steering or label
        self.filenames = []
        self.ground_truth = []
        # Determine the type of experiment (steering or collision) to compute
        # the loss
        self.exp_type = []
        for subdir in experiments:
            subpath = os.path.join(directory, subdir)
            self._decode_experiment_dir(subpath)
            
        # Conversion of list into array # nothing
        self.ground_truth = np.array(self.ground_truth, dtype = np.float64)
        self.n = self.samples
        assert self.samples > 0, "Did not find any data"

        print('Found {} images belonging to {} experiments.'.format(
                self.samples, self.num_experiments))
        

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links), key=lambda tpl: tpl[0])

    def _decode_experiment_dir(self, dir_subpath):
        # Load steerings or labels in the experiment dir
        steerings_filename = os.path.join(dir_subpath, "tracking_angle.txt")
        labels_filename = os.path.join(dir_subpath, "label.txt")
        
        # Try to load steerings first. Make sure that the steering angle or the
        # label file is in the first column. Note also that the first line are
        # comments so it should be skipped.
        try:
            ground_truth = np.loadtxt(steerings_filename, usecols=0,
                                  delimiter=',', skiprows=1)
            exp_type = 1
        except OSError as e:
            # Try load collision labels if there are no steerings
            try:
                ground_truth = np.loadtxt(labels_filename, usecols=0)
                exp_type = 0
            except OSError as e:
                print("Neither steerings nor labels found in dir {}".format(
                dir_subpath))
                raise IOError
        
        # Now fetch all images in the image subdir
        image_dir_path = os.path.join(dir_subpath, "images")

        for root, _, files in self._recursive_list(image_dir_path):
            #print('root, _, files', root, _, files)
            sorted_files = sorted(files,
                    key = lambda fname: int(re.search(r'\d+',fname).group()))
            #print('sorted_file:', sorted_files)
            for frame_number, fname in enumerate(sorted_files):
                #print('frame_number, fname:', frame_number, fname)
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(os.path.relpath(absolute_path))
                    self.ground_truth.append(ground_truth[frame_number])
                    '''
                    if self.exp_type ==1:
                        self.groundtruth[,0].append(ground_truth[frame_number])
                        self.groundtruth[,1] = 0
                    elif self.exp_type ==1:
                        self.groundtruth[,0] = 0
                        self.groundtruth[,1].append(ground_truth[frame_number])
                    '''
                    # print('self.ground_truth:', self.ground_truth)
                    
                    self.exp_type.append(exp_type)
                    self.samples += 1

    # Shuffle and split index data                 
    def pre_data(self):
        self.index_array = np.arange(self.samples)
        '''
        if self.shuffle:
            self.index_array = np.random.permutation(self.samples)
        '''
        train_val_len = int(self.samples)
        train_val_index = self.index_array
        return train_val_index


def central_image_crop(img, crop_width=150, crop_heigth=150):
    """
    Crop the input image centered in width and starting from the bottom
    in height.
    
    # Arguments:
        crop_width: Width of the crop.
        crop_heigth: Height of the crop.
        
    # Returns:
        Cropped image.
    """
    half_the_width = int(img.shape[1] / 2)
    img = img[img.shape[0] - crop_heigth: img.shape[0],
              half_the_width - int(crop_width / 2):
              half_the_width + int(crop_width / 2)]
    return img

####

class ImageDataGenerator():
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width, if < 1, or pixels if >= 1.
        height_shift_range: fraction of total height, if < 1, or pixels if >= 1.
        brightness_range: the range of brightness to apply
        shear_range: shear intensity (shear angle in degrees).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
            Points outside the boundaries of the input are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: fraction of images reserved for validation (strictly between 0 and 1).
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0., #1
                 width_shift_range=0., #2
                 height_shift_range=0., #3
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None, #4
                 preprocessing_function=None,
                 data_format='channels_last', #5
                 validation_split=0.0):
        '''
        if data_format is None:
            data_format = K.image_data_format()
        '''
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3

        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        if validation_split and not 0 < validation_split < 1:
            raise ValueError('`validation_split` must be strictly between 0 and 1. '
                             ' Received arg: ', validation_split)
        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)
        if zca_whitening:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, which overrides '
                              'setting of `featurewise_center`.')
            if featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening` '
                              'which overrides setting of'
                              '`featurewise_std_normalization`.')
        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, '
                              'which overrides setting of '
                              '`featurewise_center`.')
        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')

    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def apply_transform(self, x,
                        transform_matrix,
                        channel_axis=0,
                        fill_mode='nearest',
                        cval=0.):
        """Apply the image transformation specified by a matrix.

        # Arguments
            x: 2D numpy array, single image.
            transform_matrix: Numpy array specifying the geometric transformation.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.

        # Returns
            The transformed version of the input.
        """
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        # Arguments
            x: batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + 1e-07)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-07)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1            # 0 = 1 - 1
        img_col_axis = self.col_axis - 1            # 1 = 2 - 1
        img_channel_axis = self.channel_axis - 1    # 2 = 3 - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range: #0.2
            theta = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0

        if self.height_shift_range: #0.2
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range: #0.2
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                    [0, 1, ty],
                                    [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = self.transform_matrix_offset_center(transform_matrix, h, w)
            x = self.apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        return x

########################################################################################

class Torch_dataset_test(Dataset):
    global color_mode, crop_size, target_size
    def __init__(self, data_index, transform=None):
        self.data_index = data_index
        self.samples = len(data_index)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.groundtruth = np.zeros((dataset_test.samples, 2,), dtype = np.float64)
        # groundtruth = self.groundtruth

    def load_img(self,path, grayscale, target_size, crop_size):    
        # print(test_dir)    
        # print(path)
        img = cv2.imread(path)
        # img = cv2.imread(os.path.join(test_dir, path))
        if grayscale:
            if len(img.shape) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if target_size:
            if (img.shape[0], img.shape[1]) != target_size:
                img = cv2.resize(img, target_size)

        if crop_size:
            pass
            # img = central_image_crop(img, crop_size[0], crop_size[1])

        if grayscale:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        # Standard
        # img = (img/255.0).astype('float32')
        return np.asarray(img, dtype=np.float32) #200,200,1

    def __getitem__(self, index):
        i = index
        index_ = self.data_index[index]
        grayscale = color_mode == 'grayscale'
        center_img = self.load_img(dataset_test.filenames[index_],grayscale= grayscale,target_size=target_size, crop_size=crop_size)
        
        # center_img = ImageDataGenerator(rotation_range = 0.2, width_shift_range = 0.2, height_shift_range=0.2).random_transform(x=center_img)
        center_img = ImageDataGenerator(rescale = 1./255).standardize(x=center_img)
        center_img = self.transform(center_img)
        
        if dataset_test.exp_type[index_] == 1:
            # Steering experiment (t=1)
            self.groundtruth[i,0] = dataset_test.ground_truth[index_]
            self.groundtruth[i,1] = 0.0

        else:
            # Collision experiment (t=0)
            self.groundtruth[i,0] = 0.0
            self.groundtruth[i,1] = dataset_test.ground_truth[index_]

        # print(i, center_img.shape, 'groundtruth:', self.groundtruth[i,])
        return center_img, self.groundtruth[i,] # input: center_img (200x200x1) - output: [steer,coll] [[4x2][4x2]]
    
    def __len__(self):
        return self.samples

# Initialize model
class ResNet8(nn.Module):
    def __init__(self):
        super(ResNet8, self).__init__()
        # Input
        self.conv1 = nn.Conv2d(IN_CHANNELS, 32, kernel_size=5, stride=2, padding=5//2)
        self.mp1 = nn.MaxPool2d(3, stride=2)

        # First residual block
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=3//2)
        nn.init.kaiming_normal_(self.conv2.weight)

        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=3//2)
        nn.init.kaiming_normal_(self.conv3.weight)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=1//2)

        # Second residual block
        self.bn3 = nn.BatchNorm2d(32)
        self.act3 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight)

        self.bn4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU()
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight)

        self.conv7 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)

        # Third residual block
        self.bn5 = nn.BatchNorm2d(64)
        self.act5 = nn.ReLU()
        self.conv8 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv8.weight)

        self.bn6 = nn.BatchNorm2d(128)
        self.act6 = nn.ReLU()
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv9.weight)

        self.conv10 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        self.fc1 = nn.Linear(6272, 1) #Dense

        self.act7 = nn.ReLU()               #ReLU

        self.do1 = nn.Dropout(p=0.5)          #Dropout

        self.fc2 = nn.Linear(6272, 1)       #Dense/Final_fc1

        self.ac8 = nn.Sigmoid()
        
    def forward(self, img):        
        img = img.view(img.size(0), IN_CHANNELS, IMG_WIDTH, IMG_HEIGHT)

        x1 = self.conv1(img)
        x1 = self.mp1(x1)

        # First residual block
        x2 = self.bn1(x1)
        x2 = self.act1(x2)
        x2 = self.conv2(x2)
        
        x2 = self.bn2(x2)
        x2 = self.act2(x2)
        x2 = self.conv3(x2)
        
        x1 = self.conv4(x1)

        x3 = x1 + x2

        # Second residual block
        x4 = self.bn3(x3)
        x4 = self.act3(x4)
        x4 = self.conv5(x4)
        
        x4 = self.bn4(x4)
        x4 = self.act4(x4)
        x4 = self.conv6(x4)

        x3 = self.conv7(x3)

        x5 = x3 + x4

        # Third residual block
        x6 = self.bn5(x5)
        x6 = self.act5(x6)
        x6 = self.conv8(x6)
        
        x6 = self.bn6(x6)
        x6 = self.act6(x6)
        x6 = self.conv9(x6)

        x5 = self.conv10(x5)

        x7 = x5 + x6

        # x = x7.view(x7.shape[0], -1)      #Flatten        
        x = x7.view(x7.size(0), -1)      #Flatten        
    
        x = self.act7(x)                  # ReLU-24

        x = self.do1(x)                   # Dropout-25

        x1 = self.fc1(x)                  # Steering angle  

        x2 = self.fc2(x)                  # Collision

        x2 = self.ac8(x2)                 # Collision

        x = [x1, x2]

        return x

########################################################################################
"""
class ResNet8(nn.Module):
    def init_kernel(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)    
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize kernels of Conv2d layers as kaiming normal
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        if  isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0)
        
    def __init__(self, img_channels=1, in_height=200, in_width=200, output_dim=1):
        super(ResNet8, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,out_channels=32, 
                      kernel_size=[5,5], stride=[2,2], padding=[5//2,5//2]),
            nn.MaxPool2d(kernel_size=[3,3], stride=[2,2]))
        
        self.residual_block_1a = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3,3], 
                      padding=[3//2,3//2]))
        
        self.parallel_conv_1 = nn.Conv2d(in_channels=32,out_channels=32, 
                                         kernel_size=[1,1], stride=[2,2], 
                                         padding=[1//2,1//2])
        
        self.residual_block_2a = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=[3,3], 
                      padding=[3//2,3//2]))
        
        

        self.parallel_conv_2 = nn.Conv2d(in_channels=32,out_channels=64, 
                                         kernel_size=[1,1], stride=[2,2], 
                                         padding=[1//2,1//2])
        
        self.residual_block_3a = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=[3,3], 
                      padding=[3//2,3//2]))
        
        

        self.parallel_conv_3 = nn.Conv2d(in_channels=64,out_channels=128, 
                                         kernel_size=[1,1], stride=[2,2], 
                                         padding=[1//2,1//2])
        
        self.output_dim = output_dim

        self.last_block = nn.Sequential(
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
            # nn.Linear(6272,self.output_dim))

        self.fc1 = nn.Linear(6272, 1, bias=True) #Dense
        self.fc2 = nn.Linear(6272, 1, bias=True) #Dense

        # self.ac8 = nn.Sigmoid()
        
        '''
        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)    
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize kernels of Conv2d layers as kaiming normal
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif  isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 2)
        '''

        self.residual_block_1a.apply(ResNet8.init_kernel)
        self.residual_block_2a.apply(ResNet8.init_kernel)
        self.residual_block_3a.apply(ResNet8.init_kernel)
        self.fc2.apply(ResNet8.init_kernel)

    def forward(self, x):
        x = x.view(x.size(0), 1, 200, 200)

        x1 = self.layer1(x)
        # First residual block
        x2 = self.residual_block_1a(x1)
        x1 = self.parallel_conv_1(x1)
        x3 = x1.add(x2)
        # Second residual block
        x4 = self.residual_block_2a(x3)
        x3 = self.parallel_conv_2(x3)
        x5 = x3.add(x4)
        # Third residual block
        x6 = self.residual_block_3a(x5)
        x5 = self.parallel_conv_3(x5)
        x7 = x5.add(x6)
        
        out = x7.view(x7.size(0), -1) # Flatten
        out = self.last_block(out)

        x1 = self.fc1(out)                  # Steering angle  

        x2 = self.fc2(out)                  # Collision

        x2 = torch.sigmoid(x2)              # Collision

        x = [x1, x2]

        return x
"""
########################################################################################

class utils():

    def write_to_file(dictionary, fname):
        """
        Writes everything is in a dictionary in json model.
        """
        with open(fname, "w") as f:
            json.dump(dictionary,f)
            print("Written file {}".format(fname))

    def compute_predictions_and_gt(model, generator, steps,
                                        max_q_size=10,
                                        pickle_safe=False, verbose=0):
        """
        Generate predictions and associated ground truth
        for the input samples from a data generator.
        The generator should return the same kind of data as accepted by
        `predict_on_batch`.
        Function adapted from keras `predict_generator`.

        # Arguments
            generator: Generator yielding batches of input samples.
            steps: Total number of steps (batches of samples)
                to yield from `generator` before stopping.
            max_q_size: Maximum size for the generator queue.
            pickle_safe: If `True`, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            verbose: verbosity mode, 0 or 1.

        # Returns
            Numpy array(s) of predictions and associated ground truth.

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        """
        steps_done = 0
        all_outs = []
        all_labels = []
        all_ts = []

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = DroneDirectoryIterator.next(generator)

            if isinstance(generator_output, tuple):
                if len(generator_output) == 2:
                    x, gt_lab = generator_output
                elif len(generator_output) == 3:
                    x, gt_lab, _ = generator_output
                else:
                    raise ValueError('output of generator should be '
                                    'a tuple `(x, y, sample_weight)` '
                                    'or `(x, y)`. Found: ' +
                                    str(generator_output))
            else:
                raise ValueError('Output not valid for current evaluation')

            outs = model.predict_on_batch(x)
            if not isinstance(outs, list):
                outs = [outs]
            if not isinstance(gt_lab, list):
                gt_lab = [gt_lab]

            if not all_outs:
                for out in outs:
                # Len of this list is related to the number of
                # outputs per model(1 in our case)
                    all_outs.append([])

            if not all_labels:
                # Len of list related to the number of gt_commands
                # per model (1 in our case )
                for lab in gt_lab:
                    all_labels.append([])
                    all_ts.append([])


            for i, out in enumerate(outs):
                all_outs[i].append(out)

            for i, lab in enumerate(gt_lab):
                all_labels[i].append(lab[:,1])
                all_ts[i].append(lab[:,0])

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

        if steps_done == 1:
            return [out for out in all_outs], [lab for lab in all_labels], np.concatenate(all_ts[0])
        else:
            return np.squeeze(np.array([np.concatenate(out) for out in all_outs])).T, \
                            np.array([np.concatenate(lab) for lab in all_labels]).T, \
                            np.concatenate(all_ts[0])

# Functions to evaluate steering prediction

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)    # phuong sai
    print('Var:', np.var(y-ypred)/vary)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    """
    assert np.all(predictions.shape == real_values.shape)
    print('np.all:', np.all(predictions.shape == real_values.shape))
    ex_variance = explained_variance_1d(predictions, real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance


def compute_sq_residuals(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    sr = np.mean(sq_res, axis = -1)
    print("MSE = {}".format(sr))
    return sq_res


def compute_rmse(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values))
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse


def compute_highest_regression_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    highest_errors = sq_res.argsort()[-n_errors:][::-1]
    return highest_errors


def random_regression_baseline(real_values):
    mean = np.mean(real_values)
    std = np.std(real_values)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values):
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)

#####
def evaluate_regression(predictions, real_values, fname):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    highest_errors = compute_highest_regression_errors(predictions, real_values,
            n_errors=20)
    dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(),
                  "highest_errors": highest_errors.tolist()}
    utils.write_to_file(dictionary, fname)


# Functions to evaluate collision

def read_training_labels(file_name):
    labels = []
    try:
        labels = np.loadtxt(file_name, usecols=0)
        labels = np.array(labels)
    except:
        print("File {} failed loading labels".format(file_name)) 
    return labels


def count_samples_per_class(train_dir):
    experiments = glob.glob(train_dir + "/*")
    num_class0 = 0
    num_class1 = 0
    for exp in experiments:
        file_name = os.path.join(exp, "label.txt")
        try:
            labels = np.loadtxt(file_name, usecols=0)
            num_class1 += np.sum(labels == 1)
            num_class0 += np.sum(labels == 0)
        except:
            print("File {} failed loading labels".format(file_name)) 
            continue
    return np.array([num_class0, num_class1])


def random_classification_baseline(real_values):
    """
    Randomly assigns half of the labels to class 0, and the other half to class 1
    """
    # print([randint(0,1) for p in range(real_values.shape[0])])
    return [randint(0,1) for p in range(real_values.shape[0])]


def weighted_baseline(real_values, samples_per_class):
    """
    Let x be the fraction of instances labeled as 0, and (1-x) the fraction of
    instances labeled as 1, a weighted classifier randomly assigns x% of the
    labels to class 0, and the remaining (1-x)% to class 1.
    """
    weights = samples_per_class/np.sum(samples_per_class)
    return np.random.choice(2, real_values.shape[0], p=weights)


def majority_class_baseline(real_values, samples_per_class):
    """
    Classify all test data as the most common label
    """
    major_class = np.argmax(samples_per_class)
    return [major_class for i in real_values]

            
def compute_highest_classification_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    dist = abs(predictions - real_values)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors


def evaluate_classification(pred_prob, pred_labels, real_labels, fname):
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    precision = metrics.precision_score(real_labels, pred_labels)
    print('Precision = ', precision)
    recall = metrics.recall_score(real_labels, pred_labels)
    print('Recall = ', recall)
    f_score = metrics.f1_score(real_labels, pred_labels)
    print('F1-score = ', f_score)
    highest_errors = compute_highest_classification_errors(pred_prob, real_labels,
            n_errors=20)
    print('highest_errors:', highest_errors)
    dictionary = {"ave_accuracy": ave_accuracy.tolist(), "precision": precision.tolist(),
                  "recall": recall.tolist(), "f_score": f_score.tolist(),
                  "highest_errors": highest_errors.tolist()}
    utils.write_to_file(dictionary, fname)

##########################################################################################################

def toDevice(data, device):
    imgs, gth = data
    return imgs.float().to(device), gth.float().to(device)

model = ResNet8()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is: ', device)
model_device = model.to(device)
IN_CHANNELS = 1
IMG_WIDTH = 200
IMG_HEIGHT = 200
summary(model_device, (IN_CHANNELS, IMG_WIDTH, IMG_HEIGHT))
# Generate testing data
# test_datagen = cnn.DroneDirectoryIterator(rescale=1./255)
dataset_test = DroneDirectoryIterator(directory=test_dir)
# groundtruth = np.zeros((dataset_test.samples, 2,), dtype = np.float64)

test_index = dataset_test.pre_data()  # split data

transformations_test = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0))])
test_set = Torch_dataset_test(test_index, transformations_test)
bacth =  1 # dataset_test.samples
params = {'batch_size': bacth, 'shuffle': False, 'num_workers': 8}
test_generator = DataLoader(test_set, **params)

# print('test_generator:', test_generator)

##########################################################################################################

def difference(list1, list2):
    list_diff_1 = []
    list_diff_2 = []
    list_diff_n = []
    n = 0
    for i in list1:
        # print(n)
        if i != list2[n]:
            list_diff_1.append(list1[n])
            list_diff_2.append(list2[n])
            list_diff_n.append(n)
        n+=1
    return list_diff_1, list_diff_2, list_diff_n

def _main():
    global test_generator, PATH

    # Set testing mode (dropout/batchnormalization)
    # K.set_learning_phase(TEST_PHASE)

    # Load json and create model
    # json_model_path = os.path.join(experiment_rootdir, FLAGS.json_model_fname)
    # model = utils.jsonToModel(json_model_path)
    
    model = torch.load(PATH)

    # Compile model
    # lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # gth_test = np.zeros((dataset_test.samples, 2))
    # print(predictions)
    i = 0
    # Prediction and ground_truth
    predictions = np.zeros((dataset_test.samples, 2))
    ground_truth = np.zeros((dataset_test.samples, 2))

    # print(test_generator)
    for data_test in test_generator:
        images_test, ground_truth_test = toDevice(data_test, device)
        predict_test = model(images_test)

        predict_test_0 = predict_test[0].cpu().detach().numpy()
        predict_test_1 = predict_test[1].cpu().detach().numpy()

        predictions[i,0] = predict_test_0
        predictions[i,1] = predict_test_1
        # print('predictions:', predictions)


        ground_truth_0 = ground_truth_test[0,0].cpu().detach().numpy()
        # print('ground_truth_0:', ground_truth_0)

        ground_truth_1 = ground_truth_test[0,1].cpu().detach().numpy()
        # print('ground_truth_1:', ground_truth_1)

        ground_truth[i,0] = ground_truth_0
        ground_truth[i,1] = ground_truth_1
        # print('ground_truth:', ground_truth)
        i+=1

    # print('ground_truth:', ground_truth)
    # print('predictions:', predictions.shape)
    t = dataset_test.exp_type
    t = np.array(t)
    # print('t:', t)

    ###########################################################################

    # bug

    ###########################################################################

    # Get predictions and ground truth
    n_samples = dataset_test.samples
    print('n_samples:', n_samples)
    # batch_size = 16
    # nb_batches = int(np.ceil(n_samples / batch_size))

    # predictions, ground_truth, t = utils.compute_predictions_and_gt(model, dataset_test, nb_batches, verbose = 1)

    # Param t. t=1 steering, t=0 collision
    t_mask = t==1

    # print(t_mask)

    print('************************* Steering evaluation ***************************')
    
    # Predicted and real steerings
    pred_steerings = predictions[t_mask,0]
    # print('pred_steerings:', pred_steerings)
    real_steerings = ground_truth[t_mask,0]
    # print('real_steerings:', real_steerings)


    # Compute random and constant baselines for steerings
    random_steerings = random_regression_baseline(real_steerings)
    constant_steerings = constant_baseline(real_steerings)

    # Create dictionary with filenames
    dict_fname = {'test_regression.json': pred_steerings,
                  'random_regression.json': random_steerings,
                  'constant_regression.json': constant_steerings}

    # Evaluate predictions: EVA, residuals, and highest errors
    for fname, pred in dict_fname.items():
        abs_fname = os.path.join(experiment_rootdir, fname)
        evaluate_regression(pred, real_steerings, abs_fname)

    # Write predicted and real steerings
    dict_test = {'pred_steerings': pred_steerings.tolist(),
                 'real_steerings': real_steerings.tolist()}
    utils.write_to_file(dict_test,os.path.join(experiment_rootdir,
                                               'predicted_and_real_steerings.json'))

    print('*********************** Collision evaluation ****************************')

    # Predicted probabilities and real labels
    pred_prob = predictions[~t_mask,1]
    # print('pred_prob:', pred_prob.shape)
    pred_labels = np.zeros_like(pred_prob)
    pred_labels[pred_prob >= 0.5] = 1

    real_labels = ground_truth[~t_mask,1]
    # print('real_labels:', real_labels.shape)

    # Custom python code to check if list one is equal to list two by taking difference
    # Define function name difference

    # Initializing list 1 and list 2

    # print ("List pred: ", pred_labels)
    # print ("List real: ", real_labels)

    # Take difference of list 1 and list 2
    list_real, list_pred, list_image = difference(real_labels, pred_labels)

    # print("Difference of first and second String: ")
    # print('list_real:', list_real)
    # print('list_pred:', list_pred)
    # print('list_image:', list_image)

    filename_image = []
    for i in list_image:
        filename_image.append(dataset_test.filenames[i])

    # print('filename_image:', filename_image)

    # using naive method
    # to convert lists to dictionary
    res = {}
    n = 0
    for key in filename_image:
        res[key] = list_real[n]
        n+=1

    # Printing resultant dictionary 
    # print ("Resultant dictionary is : " +  str(res))
    
    # Compute random, weighted and majorirty-class baselines for collision
    random_labels = random_classification_baseline(real_labels)

    # Create dictionary with filenames
    dict_fname = {'test_classification.json': pred_labels,
                  'random_classification.json': random_labels}

    # Evaluate predictions: accuracy, precision, recall, F1-score, and highest errors
    for fname, pred in dict_fname.items():
        # print('fname:', fname, 'pred:', pred)
        abs_fname = os.path.join(experiment_rootdir, fname)
        evaluate_classification(pred_prob, pred, real_labels, abs_fname)

    # Write predicted probabilities and real labels
    dict_test = {'pred_probabilities': pred_prob.tolist(),
                 'real_labels': real_labels.tolist()}
    utils.write_to_file(dict_test,os.path.join(experiment_rootdir,
                                               'predicted_and_real_labels.json'))

def main(argv):
    # Utility main to load flags
    '''
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    '''
    _main()


if __name__ == "__main__":
    main(sys.argv)
