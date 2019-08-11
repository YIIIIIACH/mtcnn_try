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
  image = tf.image.resize(image, [12, 12])                              #  CONTROL THE INPUT SIZE OF MODEL
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

label_ds = tf.data.Dataset.zip((all_image_bbox_ds,all_image_landmark_ds,all_image_bbox_ds,all_image_landmark_ds,all_image_bbox_ds,all_image_landmark_ds,all_image_bbox_ds,all_image_landmark_ds,all_image_bbox_ds,all_image_landmark_ds,all_image_bbox_ds,all_image_landmark_ds,all_image_bbox_ds,all_image_landmark_ds,all_image_bbox_ds,all_image_landmark_ds,all_image_bbox_ds,all_image_landmark_ds,all_image_bbox_ds,all_image_landmark_ds))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))                         

print('image_ds',image_ds)

BATCH_SIZE = 128
image_label_ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000)) 

image_label_ds  = image_label_ds.batch(BATCH_SIZE)

image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
print('############################  image_label_ds   ######################',image_label_ds)

img_input = tf.keras.Input(shape=(12,12,3,) )
x = tf.keras.layers.Conv2D(10, (3, 3), input_shape = (12,12,3), activation = 'relu') (img_input)
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(x)
x = tf.keras.layers.Flatten()(x)
bbox_1 = tf.keras.layers.Dense(  4,name='bbox_1')(x) 
landmark_1 = tf.keras.layers.Dense(  10,name='landmark_1')(x)
bbox_2 = tf.keras.layers.Dense(  4,name='bbox_2')(x) 
landmark_2 = tf.keras.layers.Dense(  10,name='landmark_2')(x) 
bbox_3 = tf.keras.layers.Dense(  4,name='bbox_3')(x) 
landmark_3 = tf.keras.layers.Dense(  10,name='landmark_3')(x) 
bbox_4 = tf.keras.layers.Dense(  4,name='bbox_4')(x) 
landmark_4 = tf.keras.layers.Dense(  10,name='landmark_4')(x) 
bbox_5 = tf.keras.layers.Dense(  4,name='bbox_5')(x) 
landmark_5 = tf.keras.layers.Dense(  10,name='landmark_5')(x) 
bbox_6 = tf.keras.layers.Dense(  4,name='bbox_6')(x) 
landmark_6 = tf.keras.layers.Dense(  10,name='landmark_6')(x) 
bbox_7 = tf.keras.layers.Dense(  4,name='bbox_7')(x) 
landmark_7 = tf.keras.layers.Dense(  10,name='landmark_7')(x) 
bbox_8 = tf.keras.layers.Dense(  4,name='bbox_8')(x) 
landmark_8 = tf.keras.layers.Dense(  10,name='landmark_8')(x) 
bbox_9 = tf.keras.layers.Dense(  4,name='bbox_9')(x) 
landmark_9 = tf.keras.layers.Dense(  10,name='landmark_9')(x) 
bbox_10 = tf.keras.layers.Dense(  4,name='bbox_10')(x) 
landmark_10 = tf.keras.layers.Dense(  10,name='landmark_10')(x) 
 

model = tf.keras.models.Model(inputs=[img_input], outputs=[bbox_1,landmark_1,bbox_2,landmark_2,bbox_3,landmark_3,bbox_4,landmark_4,bbox_5,landmark_5,bbox_6,landmark_6,bbox_7,landmark_7,bbox_8,landmark_8,bbox_9,landmark_9,bbox_10,landmark_10])           ###########
 #  Optimizer() change ??   tf.train.AdamOptimizer()  OR   tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss={'bbox_1':'mean_squared_error','landmark_1':'mean_squared_error','bbox_2':'mean_squared_error','landmark_2':'mean_squared_error','bbox_3':'mean_squared_error','landmark_3':'mean_squared_error','bbox_4':'mean_squared_error','landmark_4':'mean_squared_error','bbox_5':'mean_squared_error','landmark_5':'mean_squared_error','bbox_6':'mean_squared_error','landmark_6':'mean_squared_error','bbox_7':'mean_squared_error','landmark_7':'mean_squared_error','bbox_8':'mean_squared_error','landmark_8':'mean_squared_error','bbox_9':'mean_squared_error','landmark_9':'mean_squared_error','bbox_10':'mean_squared_error','landmark_10':'mean_squared_error'},loss_weights={'bbox_1':0.5 , 'landmark_1':0.5 ,'bbox_2':0.5 , 'landmark_2':0.5,'bbox_3':0.5 , 'landmark_3':0.5,'bbox_4':0.5 , 'landmark_4':0.5,'bbox_5':0.5 , 'landmark_5':0.5,'bbox_6':0.5 , 'landmark_6':0.5,'bbox_7':0.5 , 'landmark_7':0.5,'bbox_8':0.5 , 'landmark_8':0.5,'bbox_9':0.5 , 'landmark_9':0.5,'bbox_10':0.5 , 'landmark_10':0.5},metrics=["accuracy"])
model.summary()

#steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

model.fit(image_label_ds, epochs=200, steps_per_epoch=50)

model.save_weights('pnet_muti_output_weight.h5')   #save our weight

