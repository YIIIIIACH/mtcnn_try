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
all_image_bbox = []
all_image_landmark = []
label_bbox_x = []
label_bbox_y = []
label_bbox_width = []
label_bbox_height = []
label_lefteye_x = []
label_lefteye_y = []
label_righteye_x = []
label_righteye_y = []
label_nose_x = []
label_nose_y = []
label_leftmouth_x = []
label_leftmouth_y = []
label_rightmouth_x = []
label_rightmouth_y = []

for i in range(0,tuple_of_data):
	all_image_paths.append(align_image_path + data[i][0]) 
	all_image_bbox.append(data[i][1:5])
	all_image_landmark.append(data[i][5:15])
#random.shuffle(all_image_paths)          C O N S I D E R I N G   T O   U S E

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [24, 24])                              #  CONTROL THE INPUT SIZE OF MODEL
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

import matplotlib.pyplot as plt


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print('THIS IS PATH_DS')
print(path_ds)

image_ds = path_ds.map(load_and_preprocess_image , num_parallel_calls=AUTOTUNE)


all_image_bbox_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_bbox, tf.int64))
all_image_landmark_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_landmark, tf.int64))


label_ds = tf.data.Dataset.zip((all_image_bbox_ds , all_image_landmark_ds))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))                         

print('image_ds',image_ds)

BATCH_SIZE = 128
image_label_ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000)) 

image_label_ds  = image_label_ds.batch(BATCH_SIZE)

image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
print('############################  image_label_ds   ######################',image_label_ds)

img_input = tf.keras.Input(shape=(24,24,3,) )
x = tf.keras.layers.Conv2D(28, (3, 3), input_shape = (24,24,3), activation = 'relu') (img_input)
x = tf.keras.layers.MaxPooling2D(pool_size = (3, 3))(x)
x = tf.keras.layers.Conv2D(48, (3, 3), activation = 'relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
x = tf.keras.layers.Conv2D(64, (2, 2), activation = 'relu')(x)
x = tf.keras.layers.Dense(128)(x)
bbox = tf.keras.layers.Dense(  4,name='bbox')(x) 
landmark = tf.keras.layers.Dense(  10,name='landmark')(x) 

model = tf.keras.models.Model(inputs=[img_input], outputs=[bbox,landmark])           ###########
 #  Optimizer() change ??   tf.train.AdamOptimizer()  OR   tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss={'bbox':'mean_squared_error','landmark':'mean_squared_error'},loss_weights={'bbox':0.5 , 'landmark':0.5 },metrics=["accuracy"])
model.summary()

#steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

model.fit(image_label_ds, epochs=10, steps_per_epoch=50)

#model.save_weights('pnet_weight.h5')   #save our weight

