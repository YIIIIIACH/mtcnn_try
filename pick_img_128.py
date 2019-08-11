import os ,sys

import numpy as np
from PIL import Image
import random
import csv
from random import *
import json

ROOT_DIR = os.getcwd()
align_image_path = os.path.join(ROOT_DIR,'img_align_celeba')
origin_image_path = os.path.join(ROOT_DIR,'img_celeba/img_celeba')
bbox_path = os.path.join(ROOT_DIR,'list_bbox_celeba.csv')
landmark_path = os.path.join(ROOT_DIR,'list_landmarks_align_celeba.csv')     #  create path

origin_image = os.listdir( origin_image_path )
oimage_amount = len(origin_image)
select_amt =  5000# oimage_amount
print('total amount of original image :',oimage_amount)
select_file = [[0] for i in range(0,select_amt)]
origin_image.sort()
print('origin_image finished sorting')
print('len(select_file) :',len(select_file))
for i in range(0,select_amt):
	select_file[i][0]= origin_image[i]  

with open(bbox_path, newline='') as bbox_file:
	# 讀取 CSV 檔案內容
	rows = list(csv.reader(bbox_file))
	
	# 以迴圈輸出每一列
	al_bbox = [0,0,0,0]
	
	for j in range(0,len(select_file)):
		#for i in range(0,len(rows)):
		#if rows[i][0] == select_file[j][0]:
		im = Image.open(origin_image_path + '/' + str(select_file[j][0]))
		al_bbox[0] = int(int(rows[j+1][1]) / im.size[0] *128)
		al_bbox[1] = int(int(rows[j+1][3]) / im.size[1] *128)
		al_bbox[2] = int(int(rows[j+1][3]) / im.size[0] *128)
		al_bbox[3] = int(int(rows[j+1][4]) / im.size[1] *128)
		select_file[j].extend(al_bbox)
		al_bbox = [0,0,0,0]
		im.close()
		#break
				
	
with open(landmark_path, newline='') as lm_file:
	# 讀取 CSV 檔案內容
	rows = list(csv.reader(lm_file))	
	# 以迴圈輸出每一列	
	for j in range(0,len(select_file)):
		#for i in range(0,len(rows)):
		#if rows[i][0] == select_file[j][0]:
		for k in range(1,6):	
			rows[j+1][2*k-1] = int(int(rows[j+1][2*k-1])*128/178)
			rows[j+1][2*k] = int(int(rows[j+1][2*k])*128/218)
						
		select_file[j].extend(rows[j+1][1:11] )
			#break
			
	
img_select_json = json.dumps(select_file)
with open('img_of_all.json', 'w') as f:
	json.dump(select_file , f)

