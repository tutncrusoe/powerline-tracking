# import libs
import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchsummary import summary
from torchvision import models
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models , transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.ndimage as ndi
import re

experiment_rootdir = '/home/rtr/Powerline_Tracking/models/'
PATH = '/home/rtr/Powerline_Tracking/models/Powerline.pth'
data_training = '/home/rtr/Powerline_Tracking/datasets/training'                # directory to data
data_validation = '/home/rtr/Powerline_Tracking/datasets/validation'                # directory to data

#classLabels = ["steering_angle", "collision"]
IN_CHANNELS = 1
IMG_WIDTH = 200
IMG_HEIGHT = 200

img_width, img_height = 320,240
crop_img_width, crop_img_height = 200,200

target_size=(320,240)
crop_size = (200,200)
color_mode = 'grayscale'
batch_size = 128

class DroneDirectoryIterator():
    global target_size , crop_size, color_mode, batch_size
    def __init__(self, directory, target_size=target_size, 
                    crop_size = crop_size, color_mode=color_mode, 
                    batch_size = batch_size, shuffle=True, seed=None, follow_links=False):
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

        # First count how many experiments are out there
        self.samples = 0

        experiments = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                experiments.append(subdir)
        self.num_experiments = len(experiments)
        self.formats = {'png', 'jpg'}
        
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

        assert self.samples > 0, "Did not find any data"

        # self.n = self.samples #1

        print('Found {} images belonging to {} experiments.'.format(
                self.samples, self.num_experiments))

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links), 
                key=lambda tpl: tpl[0])

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
            sorted_files = sorted(files,
                    key = lambda fname: int(re.search(r'\d+',fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):                       
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    # print('absolute_path:', absolute_path)
                    self.filenames.append(os.path.relpath(absolute_path,
                            self.directory))
                    # print('self.filenames:', self.filenames)
                    self.ground_truth.append(ground_truth[frame_number])
                    self.exp_type.append(exp_type)
                    self.samples += 1


    # Shuffle and split index data                 
    def pre_data(self):
        self.index_array = np.arange(self.samples)
        if self.shuffle:
            self.index_array = np.random.permutation(self.samples)
        train_val_len = int(self.samples)
        train_val_index = self.index_array
        return train_val_index

####

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

class Torch_dataset_train(Dataset):
    global color_mode, crop_size, target_size
    def __init__(self, data_index, transform = None):
        self.data_index = data_index
        self.samples = len(data_index)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def load_img(self,path, grayscale, target_size, crop_size):        
        img = cv2.imread(os.path.join(data_training, path))
        if grayscale:
            if len(img.shape) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if target_size:
            if (img.shape[0], img.shape[1]) != target_size:
                img = cv2.resize(img, target_size)

        if crop_size:
            img = central_image_crop(img, crop_size[0], crop_size[1])

        if grayscale:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        # Standard
        # img = (img/255.0).astype('float32')
        return np.asarray(img, dtype=np.float32) #200,200,1

    def __getitem__(self, index):
        index_ = self.data_index[index]
        grayscale = color_mode == 'grayscale'
        center_img = self.load_img(dataset_train.filenames[index_],grayscale= grayscale,target_size=target_size, crop_size=crop_size)
        # print('center_img:', center_img)

        center_img = ImageDataGenerator(rotation_range = 0.2, width_shift_range = 0.2, height_shift_range=0.2).random_transform(x=center_img)
        center_img = ImageDataGenerator(rescale = 1./255).standardize(x=center_img)
        center_img = self.transform(center_img)
        # print('center_img:', center_img)

        if dataset_train.exp_type[index_] == 1:
            # Steering experiment (t=1)
            steer      = np.array([1.0, dataset_train.ground_truth[index_]], dtype=np.float32)
            coll       = np.array([1.0, 0.0], dtype=np.float32)
        else:
            # Collision experiment (t=0)
            steer = np.array([0.0, 0.0], dtype=np.float32)
            coll  = np.array([0.0, dataset_train.ground_truth[index_]],dtype=np.float32)

        return center_img,[steer,coll] # input: center_img (200x200x1) - output: [steer,coll] [[4x2][4x2]]
    
    def __len__(self):
        return self.samples

class Torch_dataset_val(Dataset):
    global color_mode, crop_size, target_size
    def __init__(self, data_index, transform=None):
        self.data_index = data_index
        self.samples = len(data_index)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def load_img(self,path, grayscale, target_size, crop_size):        
        img = cv2.imread(os.path.join(data_validation, path))
        if grayscale:
            if len(img.shape) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if target_size:
            if (img.shape[0], img.shape[1]) != target_size:
                img = cv2.resize(img, target_size)

        if crop_size:
            img = central_image_crop(img, crop_size[0], crop_size[1])

        if grayscale:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        # img = (img/255.0).astype('float32')
        return np.asarray(img, dtype=np.float32)
    
    def __getitem__(self, index):
        index_ = self.data_index[index]
        grayscale = color_mode == 'grayscale'
        center_img = self.load_img(dataset_val.filenames[index_],grayscale= grayscale,target_size=target_size, crop_size=crop_size)

        center_img = ImageDataGenerator().random_transform(x=center_img)
        center_img = ImageDataGenerator(rescale = 1./255).standardize(x=center_img)
        center_img = self.transform(center_img)
        
        if dataset_val.exp_type[index_] == 1:
            # Steering experiment (t=1)
            steer      = np.array([1.0, dataset_val.ground_truth[index_]], dtype=np.float32)
            coll       = np.array([1.0, 0.0], dtype=np.float32)
        else:
            # Collision experiment (t=0)
            steer = np.array([0.0, 0.0], dtype=np.float32)
            coll  = np.array([0.0, dataset_val.ground_truth[index_]],dtype=np.float32)

        return center_img,[steer,coll] # input: center_img (200x200x1) - output: [steer,coll] [[4x2][4x2]]
    
    def __len__(self):
        return self.samples

########################################################################################



dataset_train = DroneDirectoryIterator(directory = data_training, target_size=(320,240), 
                    crop_size = (200,200), color_mode='grayscale', 
                    batch_size = 16, shuffle=True, seed=None, follow_links=False)        # load dataset

dataset_val = DroneDirectoryIterator(directory = data_validation, target_size=(320,240), 
                    crop_size = (200,200), color_mode='grayscale', 
                    batch_size = 16, shuffle=True, seed=None, follow_links=False)        # load dataset

train_index = dataset_train.pre_data()  # split data
val_index = dataset_val.pre_data()  # split data


# transformations_training = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0))])
'''
transformations_training = transforms.Compose([
    transforms.Lambda(lambda x: (x / 255.0)), transforms.RandomAffine(degrees = (-0.2, 0.2), translate = (0.2, 0.2))
])
'''
transformations_training = transforms.Compose([
    transforms.RandomAffine(degrees = (-0.2, 0.2), translate = (0.2, 0.2))
])

'''
transformations_training = transforms.Compose([
    transforms.Lambda(lambda x: (x / 255.0))
])
'''
'''
transformations_validation = transforms.Compose([
    transforms.Lambda(lambda x: (x / 255.0))
])
'''
params = {'batch_size': 16, 'shuffle': True, 'num_workers': 8}
# params = {'batch_size': 16, 'shuffle': True}

# Create data training in pytorch
# training_set = Torch_dataset_train(train_index, transformations_training)
training_set = Torch_dataset_train(train_index)
training_generator = DataLoader(training_set, **params)

# Create data validation in pytorch
# validation_set = Torch_dataset_val(val_index, transformations_validation)
validation_set = Torch_dataset_val(val_index)
validation_generator = DataLoader(validation_set, **params)

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

        self.fc1 = nn.Linear(6272, 1)       #Dense/steering_angle

        self.act7 = nn.ReLU()               #ReLU

        self.do1 = nn.Dropout(p=0.5)        #Dropout

        self.fc2 = nn.Linear(6272, 1)       #Dense/Collision

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

        ########################################        #################################

########################################################################################
alpha = Variable(torch.tensor(1).type(torch.FloatTensor), requires_grad = True)
beta = Variable(torch.tensor(0).type(torch.FloatTensor), requires_grad = True)
weights = [alpha, beta]
class_weights = torch.FloatTensor(weights).cuda()

########################################################################################
# Custom loss in torch
# Compute MSE for steering evaluation and hard-mining for the current batch
class hard_mining_mse(nn.Module):
    def __init__( self,k, **kwargs):
        self.k = k.clone().detach().to(device) # Number of samples for hard-mining
        super().__init__()
        # print('k:', k)

    def custom_mse(self,inputs, targets):
        inputs = inputs.clone().detach().requires_grad_(True)

        # Parameter t indicates the type of experiment
        t = targets[:,0]
        ones_array = torch.ones((list(t.size())[0]), dtype=torch.float64).to(device)
        # Number of steering samples
        samples_steer = torch.eq(t,ones_array).type(torch.int32)

        n_samples_steer = torch.sum(samples_steer)

        if n_samples_steer == 0:
            return 0.0
        else:
            # Predicted and real steerings
            pred_steer = torch.squeeze(inputs, -1)
            pred_steer = pred_steer.clone().detach().requires_grad_(True)
            true_steer = targets[:,1]
            # print('pred_steer:', pred_steer, '\ntrue_steer:', true_steer)

            # Steering loss
            l_steer = torch.multiply(t, torch.square(pred_steer - true_steer))
            # print('self.k:', self.k)
            # print('n_samples_steer:', n_samples_steer)

            # Hard mining
            k_min = torch.minimum(self.k, n_samples_steer)
            _, indices = torch.topk(l_steer, k = k_min.type(torch.int32))
            max_l_steer = torch.gather(l_steer,0, indices)
            hard_l_steer = torch.divide(torch.sum(max_l_steer), self.k)
    
            return hard_l_steer
    
# Compute binary cross-entropy for collision evaluation and hard-mining.
class hard_mining_entropy(nn.Module):
    def __init__( self,k, **kwargs):
        self.k = k.clone().detach().to(device) # Number of samples for hard-mining
        super().__init__()

    def custom_bin_crossentropy(self,inputs, targets):
        inputs = inputs.clone().detach().requires_grad_(True)
        bce_loss = nn.BCELoss()
        # bce_loss = nn.BCELoss(weight = torch.ones([1]).cuda())

        # Parameter t indicates the type of experiment
        t = targets[:,0]
        zeros_array = torch.zeros((list(t.size())[0]), dtype=torch.float32).to(device)
        # Number of collision samples
    
        samples_coll = torch.eq(t,zeros_array).type(torch.int32)
        n_samples_coll = torch.sum(samples_coll)
        
        if n_samples_coll == 0:
            return 0.0
        else:
            # Predicted and real labels
            pred_coll = torch.squeeze(inputs, -1)
            pred_coll = pred_coll.clone().detach().requires_grad_(True)
            true_coll = targets[:,1]
            # print('pred_coll:', pred_coll, '\ntrue_coll:', true_coll)

            # Collision loss
            l_coll = torch.multiply((1-t), bce_loss(pred_coll,true_coll))

            # Hard mining
            k_min = torch.minimum(self.k, n_samples_coll)
            _, indices = torch.topk(l_coll, k = k_min.type(torch.int32))
            max_l_coll = torch.gather(l_coll,0, indices)
            hard_l_coll = torch.divide(torch.sum(max_l_coll), self.k)
        
            return hard_l_coll
        ########################################        #################################

########################################################################################
# Load Model
model = ResNet8()
# model.load_state_dict(torch.load(PATH))
try:
    model = torch.load(PATH)
except:
    pass

# Define optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('device is: ', device)
model_device = model.to(device)
summary(model_device, (IN_CHANNELS, IMG_WIDTH, IMG_HEIGHT))

# Define loss function

batch_size = params['batch_size']
# print('batch_size:', type(batch_size))

k_mse = Variable(torch.tensor(batch_size).type(torch.Tensor), requires_grad = False)
k_entropy = Variable(torch.tensor(batch_size).type(torch.Tensor), requires_grad = False)

criterion_mse = hard_mining_mse(k_mse)
criterion_bce = hard_mining_entropy(k_entropy)

# criterion_mse = nn.MSELoss()
# criterion_bce = nn.BCELoss()

def toDevice(data, device):
    imgs, [angles, coll] = data
    return imgs.float().to(device), angles.float().to(device), coll.float().to(device)

########################################################################################             
def train_model(batch_size, model, num_epochs, learning_rate, train_loader, valid_loader, 
                patience, model_name, alpha, beta):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    k_entropy = Variable(torch.tensor(batch_size).type(torch.Tensor), requires_grad = False)

    # To track the training loss as the model trains
    # train_losses = []
    # To track the validation loss as the model trains
    # valid_losses = []
    # To track the average training loss per epoch as the model trains
    # avg_train_losses = []
    # To track the average validation loss per epoch as the model trains
    # avg_valid_losses = [] 
    
    activation_losses = []
    sigmoid_losses = []
    valid_activation_losses = []
    valid_sigmoid_losses = []
    l2_reg_losses = []

    reg_lambda = 1e-4
    # Initialize the early_stopping object
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    '''
    for i in model.named_parameters():
        if "layer_name.weight" in i:
            if l2_reg is None:
                l2_reg = i.norm(2)**2
            else:
                l2_reg = l2_reg + i.norm(2)**2
        batch_loss = some_loss_function + l2_reg * reg_lambda
    '''

    '''
    # Training loop
    decay = 1e-5 # Default 1e-5
    lambda1 = lambda step: np.exp(-(decay*step))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    '''
    # Training loop
    decay = 1e-5 # Default 1e-5
    fcn = lambda step: 1./(1. + decay*step)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fcn)
    
    lrs = []
    print('Epoch        beta            train_loss          val_loss')
    
    mse = nn.MSELoss(reduction = 'none')
    bce = nn.BCELoss(reduction = 'none')

    for epoch in range(1, num_epochs+1):
        ###################
        # TRAIN the model #
        ###################
        model.train() # prep model for training
        '''
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor])
        '''

        for batch, data_train in enumerate(train_loader, 1):
            # Load images and targets to device
            images, angles, coll = toDevice(data_train, device)
            
            # Forward pass
            outputs = model(images)
            """
            l2_reg = []
            params_dict = dict(model.named_parameters())
            for key, value in params_dict.items():
                # print(key)
                if (key == "conv2.weight" or key == "conv3.weight" or key == "conv5.weight" or key == "conv6.weight" or key == "conv8.weight" or key == "conv9.weight"):
                    '''
                    if l2_reg is None:
                        l2_reg = reg_lambda*torch.norm(value.view(value.size(0),-1),2)
                    else:
                    '''
                    # print(value.shape)
                    # l2_reg.append(reg_lambda*torch.norm(value.view(value.size(0),-1),2))
                    l2_reg.append(reg_lambda*torch.norm(value,2))

            l2_reg = sum(l2_reg)
            """
            """
            l2_reg = []
            for param in model.parameters():
                if (param == "conv2.weight" or param == "conv3.weight" or param == "conv5.weight" or param == "conv6.weight" or param == "conv8.weight" or param == "conv9.weight"):
                    l2_reg.append(reg_lambda*0.5*param.norm(2)**2)

            l2_reg = sum(l2_reg)
            """

            # Calculate loss
            ###################

            # Parameter t indicates the type of experiment
            t = angles[:,0]
            # print('t_mse:', t)
            ones_array = torch.ones((list(t.size())[0]), dtype=torch.float64).to(device)
            # print('ones_array_mse:', ones_array)
            # Number of steering samples
            samples_steer = torch.eq(t,ones_array).type(torch.int32)
            # print('samples_steer:', samples_steer)
            n_samples_steer = torch.sum(samples_steer)
            # print('n_samples_steer', n_samples_steer)

            if n_samples_steer == 0:
                loss_1 = 0.0
            else:
                # Predicted and real steerings
                pred_steer = torch.squeeze(outputs[0], -1)
                true_steer = angles[:,1]

                # Hard mining
                l_steer = torch.multiply(t, torch.square(pred_steer - true_steer))
                # print('l_steer:', l_steer)

                # Hard mining
                k_min_steer = torch.minimum(k_mse.to(device), n_samples_steer)
                # print('k_min_steering:', k_min_steer)

                _, indices = torch.topk(l_steer, k = k_min_steer.type(torch.int32))
                # Position of each sample
                # print('indices:', indices)
                max_l_steer = torch.gather(l_steer,0, indices)
                # print('max_l_steer:', max_l_steer)
                loss_1 = torch.divide(torch.sum(max_l_steer), k_mse)

            ###################

            # Parameter t indicates the type of experiment
            t = coll[:,0]
            # print('t_bce:', t)

            zeros_array = torch.zeros((list(t.size())[0]), dtype=torch.float32).to(device)
            # Number of collision samples
            # print('zeros_array_coll:', zeros_array)
            samples_coll = torch.eq(t,zeros_array).type(torch.int32)
            # print('samples_coll:', samples_coll)
            n_samples_coll = torch.sum(samples_coll)
            # print('n_samples_coll', n_samples_coll)
            
            if n_samples_coll == 0:
                loss_2 = 0.0
            else:
                # Predicted and real labels
                pred_coll = torch.squeeze(outputs[1], -1)
                true_coll = coll[:,1]

                l_coll = torch.multiply((1-t), bce(pred_coll,true_coll))
                # print('bceloss:', bce_loss(pred_coll,true_coll))
                # print('l_coll:', l_coll)

                # Hard mining
                k_min = torch.minimum(k_entropy.to(device), n_samples_coll)
                # print('k_min_coll:', k_min)
                _, indices = torch.topk(l_coll, k = k_min.type(torch.int32))
                # print('indices:', indices)
                max_l_coll = torch.gather(l_coll,0, indices)
                # print('max_l_coll:', max_l_coll)
                loss_2 = torch.divide(torch.sum(max_l_coll), k_entropy)

            loss = loss_1*alpha + loss_2*beta 
            # loss = loss_1 + loss_2

            reg_loss = 0
            params_dict = dict(model.named_parameters())
            for key, value in params_dict.items():
                # print('#################################')
                # print(key, value)
                # print('#################################')

                if (key == "conv2.weight" or key == "conv3.weight" or key == "conv5.weight" or key == "conv6.weight" or key == "conv8.weight" or key == "conv9.weight"):
                    # print(key, value)

                    # reg_loss += reg_lambda*torch.norm(value.view(value.size(0),-1),2)
                    # value = Variable(value.type(torch.FloatTensor), requires_grad = True)
                    # reg_loss += reg_lambda*(value**2).sum()
                    reg_loss += reg_lambda*((value.norm(2))**2)

                    # print(value.shape)
                    # l2_reg.append(reg_lambda*torch.norm(value.view(value.size(0),-1),2))
                    # l2_reg.append(reg_lambda*torch.norm(value,2))
            '''
            reg_loss = 0
            for key, value in model.parameters():
                if (key == "conv2.weight" or key == "conv3.weight" or key == "conv5.weight" or key == "conv6.weight" or key == "conv8.weight" or key == "conv9.weight"):
                    if reg_loss == 0:
                        reg_loss = 0.5 * (value ** 2).sum()
                        
                    else:
                        # reg_loss = reg_loss + 0.5 * param.norm(2)**2
                        reg_loss = reg_loss + 0.5 * (value ** 2).sum()
            '''
            # print("{:.10f}".format(reg_loss))
            loss += reg_loss
            # print(l2_reg)

            # Clear gradients
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            
            lrs.append(optimizer.param_groups[0]["lr"])

            # Decay Learning Rate     
            scheduler.step()

            # Record training loss
            activation_losses.append(loss_1)
            sigmoid_losses.append(loss_2)
            l2_reg_losses.append(reg_loss)
            # print('######################')
            
            # print('Train Loss: %.4f - mse_loss: %.4f - bce_loss: %.4f' % (loss, loss_1, loss_2))


        ######################    
        # VALIDATE the model #
        ######################
        
        # with torch.set_grad_enabled(False):
        with torch.no_grad():
            model.eval() # prep model for evaluation

            for data_val in valid_loader:
                images_val, angles_val, coll_val = toDevice(data_val, device)
                
                # Forward pass:
                outputs_val = model(images_val)

                # Calculate loss
                ###################

                # Parameter t indicates the type of experiment
                t = angles_val[:,0]
                # print('t_mse:', t)
                ones_array = torch.ones((list(t.size())[0]), dtype=torch.float64).to(device)
                # print('ones_array_mse:', ones_array)
                # Number of steering samples
                samples_steer = torch.eq(t,ones_array).type(torch.int32)
                # print('samples_steer:', samples_steer)
                n_samples_steer = torch.sum(samples_steer)
                # print('n_samples_steer', n_samples_steer)

                if n_samples_steer == 0:
                    loss_1_val = 0.0
                else:
                    # Predicted and real steerings
                    pred_steer = torch.squeeze(outputs_val[0], -1)
                    true_steer = angles_val[:,1]
                    
                    l_steer = torch.multiply(t, torch.square(pred_steer - true_steer))
                    # print('l_steer:', l_steer)

                    # Hard mining
                    k_min_steer = torch.minimum(k_mse.to(device), n_samples_steer)
                    # print('k_min_steering:', k_min)

                    _, indices = torch.topk(l_steer, k = k_min_steer.type(torch.int32))
                    # Position of each sample
                    # print('indices:', indices)
                    max_l_steer = torch.gather(l_steer,0, indices)
                    # print('max_l_steer:', max_l_steer)
                    loss_1_val = torch.divide(torch.sum(max_l_steer), k_mse)

                ###################

                # Parameter t indicates the type of experiment
                t = coll_val[:,0]
                # print('t_bce:', t)

                zeros_array = torch.zeros((list(t.size())[0]), dtype=torch.float32).to(device)
                # Number of collision samples
                # print('zeros_array_coll:', zeros_array)
                samples_coll = torch.eq(t,zeros_array).type(torch.int32)
                # print('samples_coll:', samples_coll)
                n_samples_coll = torch.sum(samples_coll)
                # print('n_samples_coll', n_samples_coll)

                if n_samples_coll == 0:
                    loss_2_val = 0.0
                else:
                    # Predicted and real labels
                    pred_coll = torch.squeeze(outputs_val[1], -1)
                    true_coll = coll_val[:,1]

                    l_coll = torch.multiply((1-t), bce(pred_coll,true_coll))
                    # print('bceloss:', bce_loss(pred_coll,true_coll))
                    # print('l_coll:', l_coll)

                    # Hard mining
                    k_min_coll = torch.minimum(k_entropy.to(device), n_samples_coll)
                    # print('k_min_coll:', k_min)
                    _, indices = torch.topk(l_coll, k = k_min_coll.type(torch.int32))
                    # print('indices:', indices)
                    max_l_coll = torch.gather(l_coll,0, indices)
                    # print('max_l_coll:', max_l_coll)
                    loss_2_val = torch.divide(torch.sum(max_l_coll), k_entropy)

                # loss_val = loss_1_val*alpha + loss_2_val*beta
                # loss_val = loss_1_val + loss_2_val

                # valid_losses.append(loss_val.item())

                # loss_1_val = loss_1_val.cpu().detach().numpy()
                # loss_2_val = loss_2_val.cpu().detach().numpy()
                valid_activation_losses.append(loss_1_val)
                valid_sigmoid_losses.append(loss_2_val)
                '''
                print('k_mse:', k_mse)
                print('k_entropy:', k_entropy)
                print('l_coll:', l_coll)
                print('k_min_coll:', k_min_coll)
                print('indices:', indices)
                print('max_l_coll:', max_l_coll)
                print('loss_2_val:', loss_2_val)

                print('###################')
                '''
            # Print training/validation statistics 
            # Calculate average loss over an epoch

            activation_loss = sum(activation_losses) / len(activation_losses)
            sigmoid_loss = sum(sigmoid_losses) / len(sigmoid_losses)
            l2_reg_loss = sum(l2_reg_losses) / len(l2_reg_losses)
            train_loss = alpha*activation_loss + beta*sigmoid_loss + reg_loss

            valid_activation_loss = sum(valid_activation_losses) / len(valid_activation_losses)
            valid_sigmoid_loss = sum(valid_sigmoid_losses) / len(valid_sigmoid_losses)
            valid_loss = alpha*valid_activation_loss + beta*valid_sigmoid_loss + reg_loss

            valid_ex = valid_activation_loss + valid_sigmoid_loss
            
            epoch_len = len(str(num_epochs))
            
            print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' + 
                        f'beta: {beta:.5f} ' +
                        f'| train_loss: {train_loss:.5f} ' + f'activation_loss: {activation_loss:.5f} ' + f'sigmoid_loss: {sigmoid_loss:.5f} ' + 
                        f'| valid_loss: {valid_loss:.5f} ' + f'valid_activation_loss: {valid_activation_loss:.5f} ' + f'valid_sigmoid_loss: {valid_sigmoid_loss:.5f} ' +
                        f'| valid_ex: {valid_ex:.5f} ' + f'| l2_reg_loss: {l2_reg_loss:.5f} '
                        #  f'k_mse: {k_mse:.5f} ' + f'k_entropy: {k_entropy:.5f} '
                        )
            
            # print_msg = (f'{train_loss:.5f}' + f'   {valid_loss:.5f}')
            print(print_msg)

            # Clear lists to track next epoch
            activation_losses = []
            sigmoid_losses = []
            valid_activation_losses = []
            valid_sigmoid_losses = []
            l2_reg_losses = []
            # Early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            # early_stopping(valid_loss, model)
            '''
            if early_stopping.early_stop:
                print("Early stopping")
                break    
            '''
            beta = np.maximum(0.0, 1.0-np.exp(-1.0/10.0*(epoch-10)))
            
            mse_function = batch_size-(batch_size-10)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0))))
            entropy_function = batch_size-(batch_size-5)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0))))
            
            k_mse = int(np.round(mse_function))
            k_entropy = int(np.round(entropy_function))

            k_mse = Variable(torch.tensor(k_mse).type(torch.Tensor), requires_grad = False)
            k_entropy = Variable(torch.tensor(k_entropy).type(torch.Tensor), requires_grad = False)
            
    # Load the last checkpoint with the best model 
    # (returned by early_stopping call)

        torch.save(model, os.path.join(experiment_rootdir, 'epoch-{}-val_loss-{}.pth'.format(epoch, valid_loss)))

    # model.load_state_dict(torch.load('checkpoint.pt'))
    print('Training completed and model saved.')
    plt.plot(lrs)
    plt.show()
    return  model #, avg_train_losses, avg_valid_losses

if __name__=="__main__":
    train_model(batch_size=128, model=model_device, num_epochs=150, learning_rate=0.001, train_loader=training_generator, valid_loader=validation_generator, 
                    patience=0, model_name= 'resnet8', alpha=1, beta=0)

    PATH = experiment_rootdir + '/model_Powerline_Tracking.pth'

    torch.save(model, PATH)