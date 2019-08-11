import tensorflow as tf
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import os
import random
import json
from os import listdir
from os.path import isfile, join


data_root = pathlib.Path('/home/yiiiiiach/MTCNN_TRY/image_align_celeba/')

ROOT_DIR = os.getcwd()
align_image_path = os.path.join(ROOT_DIR,'img_align_celeba/')

with open('./img_of_all.json', 'r') as f:
	data = json.load(f)	
	tuple_of_data = len(data)	
	print('tuple_of_data',tuple_of_data )

all_image_paths = []

for i in range(0,tuple_of_data):
	all_image_paths.append(align_image_path + data[i][0]) 

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [12, 12])                              #  CONTROL THE INPUT SIZE OF MODEL
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

import matplotlib.pyplot as plt


img_input = tf.keras.Input(shape=(12,12,3,) )
x = tf.keras.layers.Conv2D(10, (3, 3), input_shape = (12,12,3), activation = 'relu') (img_input)
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(x)
x = tf.keras.layers.Flatten()(x)
bbox = tf.keras.layers.Dense(  4,name='bbox')(x) 
landmark = tf.keras.layers.Dense(  10,name='landmark')(x) 

model = tf.keras.models.Model(inputs=[img_input], outputs=[bbox,landmark])           ###########
 #  Optimizer() change ??   tf.train.AdamOptimizer()  OR   tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss={'bbox':'mean_squared_error','landmark':'mean_squared_error'},loss_weights={'bbox':0.5 , 'landmark':0.5 },metrics=["accuracy"])
model.summary()

#steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
model.load_weights('pnet_weight.h5')
test_image_path = all_image_paths[random.randint(0,tuple_of_data)]                  # the file to be test
list_img = []
list_img.append(test_image_path)

print('test_image_path', test_image_path)

test_path = tf.data.Dataset.from_tensor_slices(list_img )

test_image = test_path.map(load_and_preprocess_image , num_parallel_calls=AUTOTUNE)
test_image = test_image.batch(1)

result = model.predict(test_image)
print('-------------------------------------------------------------------------')
print('\n\n\n')
print('test_image_path', test_image_path)
print('result =======>> :   ', result)



