# Animal Recognition in Tensorflow

* This is a general guide with notes on the implementation of Computer Vision applications via Tensorflow and Python, the dataset I am using for the purposes of this project can be found [here](https://www.kaggle.com/alessiocorrado99/animals10#OIP---lAIbDlHKmejDpqrXq6vAAAAA.jpeg).

* I wrote this in Jupyter Notebook to make it easier to take notes and make observations especially when implementing the more complicated Inception Networks, but one thing important to note is that the enviornment that Jupyter runs in does not allow for direct hardware access.
    * An interesting fact about Tensorflow that most people know, is that the core feature of Tensorflow involves the creation of a computation graph which is deployed via C++ in the background to the GPU. Which cannot be done in Jupyter
    * When you see the **.fit** method used anywhere and don't see any output it is for this reason, compile this program on your IDE of choice and run .fit there, Jupyter can't access the GPU --> It can actually access the GPU if you install tensorflow-gpu, but will run poorly on a Mac.

#### Goals of this project:
0. [Image loading from directories](#0.-Image-Loading-from-Directories)
<br/><br/>
1. [Pre-processing and Data Augmentation](#1.-Pre-processing-and-Data-Augmentation)
<br/><br/>
2. [Experiment with ILSVRC pre-trained models](#2.-Experiment-with-ILSVRC-pre-trained-models)
<br/><br/>
3. [Experiment with custom model development](#3.-Experiment-with-custom-model-development)
    * Different optimizers
    * loss functions
    * callbacks
<br/><br/>
4. [Tensorboard](#4.-Tensorboard)
    * What is it? How does it work? Why use it?
<br/><br/>
5. [Exporting a model](#5.-Exporting-a-model)
    * TF-Slim
    * h5
    * Tensorflow Model Save/Load procedure


```python
directory = 'animals' # Directory of the animal images
animals = ['cat', 'butterfly', 'dog', 'sheep', 'spider', 'chicken', 'horse', 'squirrel', 'cow', 'elephant']
num_classes = len(animals)

# Tensorflow version I plan on using for this project
import tensorflow as tf
print(tf.__version__)
```

    2.1.0


## 0. Image Loading from Directories

### Loading Images using Pillow


```python
from PIL import Image
import IPython.display as display
import os

file_paths = [] # file_paths for targets
targets = [] # targets

# Parse animal folder and set file_path and targets arrays
for animal in animals:
    animal_dir = os.path.join(directory,animal)
    for file in os.listdir(animal_dir):
        file_paths.append(os.path.join(animal_dir, file))
        targets.append(animals.index(animal))

print('Picture of', animals[targets[0]])
display.display(Image.open(file_paths[0]))
```

    Picture of cat



![png](output_3_1.png)


* This is great, but it was a lot of work to do all of this manually, Keras and Tensorflow make it easy

### Loading Images using Keras
* Keras comes with the ability to load images and the Tensorflow website [here](https://www.tensorflow.org/tutorials/load_data/images) has a bunch of great examples on implementation


```python
import tensorflow as tf

import numpy as np
# The 1./255 is to convert from uint8 to float32 in range [0,1]
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

image_count = len(file_paths)
BATCH_SIZE = 32 # 32 for visualization, but 256 is typical for large scale image applications
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = img_gen.flow_from_directory(directory=str(directory), 
                                             follow_links=True,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = (animals))
```

    Found 26179 images belonging to 10 classes.


The code above is pretty straightforward, the keras ImageDataGenerator utility provides an easy way to rescale and load images from paths. However one thing I do want to note is a very useful feature for directories with hierarchies.

In the event that you have folders within folders containing images, i.e: animals/cats/... and animals/dogs/..., you want to pay attention to:

**follow_links = True**

By default this is False, but setting it to True allows the DataGenerator to automatically loop through each individual directory according to the classes list


```python
import matplotlib.pyplot as plt

# You need to turn the classes into a numpy array
animal_np = np.array(animals)

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(animal_np[label_batch[n]==1][0].title())
      plt.axis('off')

image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)
```

This method through Keras is great, but has a ton of problems:
1. Slow
2. No fine-grain control
3. Not integrated with TF directly (harder to troubleshoot when plugging into ConvNet)

### Loading images using tf.data


```python
list_ds = tf.data.Dataset.list_files(str(directory + '/*/*')) # /*/* go down to the files
for f in list_ds.take(5):
  print(f.numpy())
```

    b'animals/chicken/OIP-ipmrdC3vai2nRq29JjRRYAHaFj.jpeg'
    b'animals/spider/OIP-68GOEkKdJausgJ9Jn9ng2QHaE7.jpeg'
    b'animals/chicken/OIP-9pI6oiTbvQWEk4j3LEd3vAHaHt.jpeg'
    b'animals/cat/980.jpeg'
    b'animals/spider/OIP-vEoiZNkceJgW7PEjWswAvAHaIM.jpeg'



```python
def decode_img(img):
  # convert the raw string into a 3d tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def get_label_image_pair(file_path):
    
    # Find the class name -----------------------------
    segments = tf.strings.split(file_path, os.path.sep)
    # The second to last is the directory name
    tensor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mask = segments[-2] == animals
    label = tf.boolean_mask(tensor, mask) # CONVERT TO ONE-HOT
    
    # Get the image in raw format ---------------------
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

labeled_ds = list_ds.map(get_label_image_pair) #num_parallel_calls=tf.data.experimental.AUTOTUNE)

labeled_ds = labeled_ds.shuffle(buffer_size=1000).batch(32)
    
for image, label in labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", len(label.numpy()), label.dtype)
```

    Image shape:  (32, 224, 224, 3)
    Label:  32 <dtype: 'int32'>


I'll be using the tensorflow.data method of loading images in order to make the process of image augmentation easier

### Constructing a dataset from python lists
* In order to simplify development, I will be using the MNIST recognition dataset in python which can be swapped out for the appropriate dataset in production.



```python
from tensorflow import keras
import tensorflow as tf

# Use the fashion dataset
fashion_mnist = keras.datasets.fashion_mnist

# Get the images and labels from keras
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# In order to input the mnist images, you need to reshape the training data (look below for more info)
train_images = train_images.reshape(-1,28, 28, 1)

# Create a dataset from the appropriate data
x = tf.data.Dataset.from_tensor_slices(train_images)
y = tf.data.Dataset.from_tensor_slices(train_labels)

# Zip the dataset together (creating an x and y)
mnist_dataset = tf.data.Dataset.zip((x, y))
# Shuffle the data and put it into batches
mnist_dataset = mnist_dataset.shuffle(buffer_size=1000).batch(32)
```

* One thing that might not be clear here is why you need to reshape the image data:
    * The first (-1) identifies the image index in the batch while the last (1) is to add a bias value for training and predictions

## 1. Pre-processing and Data Augmentation
* This still needs to be completed

## 2. Experiment with ILSVRC pre-trained models
* Now we have a completed dataset, the next step is to mess around with some of the basic models from the ImageNet Large Scale Visual Recognition Challenge, a competition that standardized basic CV models for use in the industry. It ran until the models produced achieved accuracy better than a human.

Loading the ILSVRC models in Tensorflow has been made really easy with the model downloading features introduced to TF. 

* The goals of this section of the project are as follows:
    * Download multiple popular ILSVRC models
    * Experiment with fitting the data
    * Experiment with different compilers and loss functions + trade-offs
    * Analyze model architecture
    
### Downloading the models
#### VGG - Visual Geometry Group @ Oxford (2014 2nd place)
* Made use of **ReLU** (Rectified Linear Unit) as activation function common in CNNs for adding non-linearity
* Applied **dropout** to CNN architecture
* Standardized structure of CNN layers --> Dense (Fully-Connected) layers


```python
vgg_net = tf.keras.applications.VGG16(
    include_top = True, weights=None, # you can set weights to 'imagenet'
    input_tensor=None, input_shape = None, pooling = None, 
    classes = 10 # 1000 for imagenet weights
)

vgg_net.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])
```


```python
print(vgg_net.summary())
```

    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc1 (Dense)                  (None, 4096)              102764544 
    _________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312  
    _________________________________________________________________
    predictions (Dense)          (None, 10)                40970     
    =================================================================
    Total params: 134,301,514
    Trainable params: 134,301,514
    Non-trainable params: 0
    _________________________________________________________________
    None



```python
# vgg_net.fit(labeled_ds, epochs=200, verbose=1)
```

## 3. Experiment with custom model development
* For the purposes of simplifying development of the model and having the ability to finish training in a reasonable time on my computer, I will use the MNIST dataset that was created above. The Animal dataset can be run in the python file located in the PyCharm directory on your own computer with the model of your choice.

* Goals in this portion of the project
    * [Different optimizers](#Optimizers)
    * loss functions
    * callbacks


```python
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input

input_shape = (28,28,1) # Input shape is taken from the MNIST dataset, alter it for animal images
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(1,1), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(1,1), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(68, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

```


```python
for image, label in mnist_dataset.take(1):
    print("Image shape: ", image.numpy().shape, ' --> (batch_size, image_size[0], image_size[1], bias_size})')
    print("Label: ", label.numpy(), label.dtype)
```

    Image shape:  (32, 28, 28, 1)  --> (batch_size, image_size[0], image_size[1], bias_size})
    Label:  [8 7 0 6 2 0 0 6 4 7 1 1 6 4 7 2 9 6 2 4 3 2 1 5 9 5 5 9 7 5 7 9] <dtype: 'uint8'>


* Now we can fit the model to our data


```python
model.fit(mnist_dataset, epochs=5, verbose=1)
```

    Train for 1875 steps
    Epoch 1/5
    1875/1875 [==============================] - 63s 33ms/step - loss: 0.3862 - accuracy: 0.8621
    Epoch 2/5
    1875/1875 [==============================] - 59s 31ms/step - loss: 0.3646 - accuracy: 0.8701
    Epoch 3/5
    1875/1875 [==============================] - 226s 121ms/step - loss: 0.3374 - accuracy: 0.8790
    Epoch 4/5
    1875/1875 [==============================] - 70s 37ms/step - loss: 0.3185 - accuracy: 0.8845
    Epoch 5/5
    1875/1875 [==============================] - 69s 37ms/step - loss: 0.3057 - accuracy: 0.8889





    <tensorflow.python.keras.callbacks.History at 0x102f92dd0>



![verbosity](verbosity.png)


```python
print(model.summary()) # We can get a description of the model after training
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_5 (Conv2D)            (None, 24, 24, 32)        832       
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 22, 22, 32)        9248      
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 22, 22, 32)        1056      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 11, 11, 32)        0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 9, 9, 32)          9248      
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 9, 9, 32)          1056      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 4, 4, 32)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 512)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               65664     
    _________________________________________________________________
    dense_1 (Dense)              (None, 68)                8772      
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                690       
    =================================================================
    Total params: 96,566
    Trainable params: 96,566
    Non-trainable params: 0
    _________________________________________________________________
    None


* An accuracy of 89% is not awful, but we can improve this using optimizers, loss functions, and callbacks (which is related to the deployment of Tensorboard in section 4)

### Optimizers <Page 100>
* As we all learned in Data Science, Optimizers are a critical part of any adaptive algorithm, there are a few common Optimizers used in Computer Vision that can enable us to develop better applications, namely:
    * 
    
### Loss Functions https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
*https://arxiv.org/abs/1511.08861

### Callbacks https://keras.io/callbacks/


## 4. Tensorboard
* Tensorboard is a visualization and measurement tool to help the developer improve training, you can read the full docs [here](https://www.tensorflow.org/tensorboard/get_started), but this section will focus on it's integration to a Tensorflow project in python.



```python
# Load the TensorBoard in Notebook
%load_ext tensorboard
```

    The tensorboard extension is already loaded. To reload it, use:
      %reload_ext tensorboard



```python
# The directory where you want your logging to be stored --> 
# Create the directory here, 'fit' is b/c this is for training
import datetime
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# You need to add a callback while training in order to observe fitting process in Training
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
%tensorboard --logdir logs/fit

```


    Reusing TensorBoard on port 6006 (pid 12836), started 0:05:33 ago. (Use '!kill 12836' to kill it.)




<iframe id="tensorboard-frame-789505ce848c7bc1" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-789505ce848c7bc1");
    const url = new URL("/", window.location);
    url.port = 6006;
    frame.src = url;
  })();
</script>



## 5. Exporting a model


```python

```
