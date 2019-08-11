import tensorflow as tf
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib

data_root = pathlib.Path('/home/yiiiiiach/MTCNN_TRY/image')

print(data_root)
import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
image_count
import os

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)  							 # >>>['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

label_to_index = dict((name, index) for index,name in enumerate(label_names))
print(label_to_index)  							#>>>{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths] 			#>>>[1, 2, 2, 4, 1, 4, 2, 1, 1, 1]

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [12, 12])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

import matplotlib.pyplot as plt
'''
img_path = all_image_paths[0]
label = all_image_labels[0]
'''

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print('THIS IS PATH_DS')
print(path_ds)

image_ds = path_ds.map(load_and_preprocess_image , num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
label_ds = tf.data.Dataset.zip( (label_ds, label_ds))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))                         
print(image_label_ds)


BATCH_SIZE = 32
image_label_ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))  
image_label_ds  = image_label_ds.batch(BATCH_SIZE)
image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)


#------------------------------------------------------------------------------------------------------------------------------------------------
img_input = tf.keras.Input(shape=(12,12,3,) )
x = tf.keras.layers.Conv2D(10, (3, 3), input_shape = (12,12,3), activation = 'relu') (img_input)
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(x)
x = tf.keras.layers.Flatten()(x)
output_1 = tf.keras.layers.Dense( len(label_names) ,name='output_1')(x)
output_2 = tf.keras.layers.Dense( len(label_names) ,name='output_2' )(x)

model = tf.keras.models.Model(inputs=[img_input], outputs=[output_1,output_2])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss={'output_1': 'categorical_crossentropy','output_2': 'categorical_crossentropy'},
              loss_weights={'output_1': 0.8, 'output_2': 0.2},
              metrics=["accuracy"])

model.summary()
#steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

model.fit(image_label_ds, epochs=10, steps_per_epoch=50)

#model.save_weights('exp_load_img_cnn_weight.h5')   #save our weight


