import tensorflow as tf
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import os
import random
import json
from os import listdir
from os.path import isfile, join
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											# # # #											
											#PART 1#
											# # # # 
data_root = pathlib.Path('/home/yiiiiiach/MTCNN_TRY/image_align_celeba/')          #  打開  ./img_of_all.json 、image directory 將所有 landmark 與 bbox 標注 、圖片path 
										   #  儲存到 1. all_image_paths  2. all_image_bbox 3. all_image_landmark 三個list 中

ROOT_DIR = os.getcwd()
align_image_path = os.path.join(ROOT_DIR,'img_align_celeba/')

with open('./img_of_all.json', 'r') as f:
	data = json.load(f)	
	tuple_of_data = len(data)	
	print('tuple_of_data',tuple_of_data )


all_image_paths = []
all_image_bbox = []
all_image_landmark = []


for i in range(0,tuple_of_data):
	all_image_paths.append(align_image_path + data[i][0]) 
	all_image_bbox.append(data[i][1:5])
	all_image_landmark.append(data[i][5:15])
#random.shuffle(all_image_paths)          C O N S I D E R I N G   T O   U S E
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											# # # #											
											#PART 2#
											# # # # 
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)											
  image = tf.image.resize(image, [48, 48])                             		#宣告能讀取圖片並將像素轉為適合 tensorflow.Data  Class 的格式  的函式
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											# # # #											
											#PART 3#
											# # # # 
import matplotlib.pyplot as plt


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)               #  tf.data.Dataset.from_tensor_slices(  <list>  ) ==> 將list 中的資料切成片段 以方便以類似串流的方式 丟進模型裡訓練
image_ds = path_ds.map(load_and_preprocess_image , num_parallel_calls=AUTOTUNE) 
								            #  tf.data.Dataset.map(  <function_name>  )  ==> 將 Class Dataset 裡面的資料使用function操作
									    #  tf.cast ( <object> , <dtype> )  ==> 轉換資料格式
all_image_bbox_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_bbox, tf.int64))
all_image_landmark_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_landmark, tf.int64))


label_ds = tf.data.Dataset.zip((all_image_bbox_ds , all_image_landmark_ds))#  tf.data.Dataset.zip(( <Dataset> , <Dataset> , ........... ))   將Dataset 結合在一起 (要注意是否與 model 合適）
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))                 #  image_label_ds 結構解析 ==>>
									#     [  [image]  , [ bbox ,  landmark ]  ]
									#        輸入         label_1  label_2
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											# # # #											
											#PART 4#
											# # # # 
print('image_ds',image_ds)

BATCH_SIZE = 128
image_label_ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))    # 將 image_label_ds 進行 1. 重排 2. batch化  等預處理
image_label_ds  = image_label_ds.batch(BATCH_SIZE)
image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)


img_input = tf.keras.Input(shape=(48,48,3,) )                                                      #  tf.keras.Input( ）宣告 input 的 shape 
										      #--------------------------------------------------------#
x = tf.keras.layers.Conv2D(32, (3, 3), input_shape = (48,48,3), activation = 'relu') (img_input)   #  keras function api 方法 建構 model 
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size = (3, 3))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
x = tf.keras.layers.Conv2D(128, (2, 2), activation = 'relu')(x)
x = tf.keras.layers.Dense(256)(x)
bbox = tf.keras.layers.Dense(  4,name='bbox')(x)                                                 # 為了產生多個輸出 宣告兩個 dense layer 分別輸出 bbox  , landmark
landmark = tf.keras.layers.Dense(  10,name='landmark')(x) 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											# # # #											
											#PART 5#
											# # # # 
model = tf.keras.models.Model(inputs=[img_input], outputs=[bbox,landmark])                    #宣告model 的輸入與輸出 對應到哪個 layer  , 其中 tf.keras.models.Model() 的參數 要與輸入輸出layer 的name 一樣
#  Optimizer() change ??   tf.train.AdamOptimizer()  OR   tf.keras.optimizers.RMSprop(0.001)   #因為輸出為 regression 故使用 RMSoptimizer      
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss={'bbox':'mean_squared_error','landmark':'mean_squared_error'},loss_weights={'bbox':0.5 , 'landmark':0.5 },metrics=["accuracy"])
											      # 宣告 每個輸出使用的loss function 以及各自權重 、 可以發現到其 資料結構與當初 宣告 image_label_ds 結構呼應
model.summary()										      # echo model 的架構與參數量

#steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

model.fit(image_label_ds, epochs=10, steps_per_epoch=50)                                    # 丟入 model 訓練

model.save_weights('pnet_weight.h5')   #save our weight

