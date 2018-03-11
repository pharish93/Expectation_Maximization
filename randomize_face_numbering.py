# Harish - 7th Feb - Program to randomly read file
# names and update them into a folder sequentially  named

import os, random
import cv2
# input_path = "./positive_faces/"
# count = 0
# while os.listdir(input_path)!=[]:
#     k = random.choice(os.listdir(input_path))
#     input_file_name = input_path + k
#     img = cv2.imread(str(input_file_name))
#     # img_resize = cv2.resize(img,(60,60))
#     img_resize = img
#     str1 = './renamed_positive_faces/positive_face_' + str(count) + '.png'
#     cv2.imwrite(str1,img_resize)
#     count = count+1
#     os.remove(input_file_name)

count = 0
input_path = "./dinesh_face/"
while os.listdir(input_path)!=[]:
    k = random.choice(os.listdir(input_path))
    input_file_name = input_path + k
    img = cv2.imread(str(input_file_name))
    # img_resize = cv2.resize(img,(60,60))
    img_resize = img
    str1 = './new/positive_face_' + str(count) + '.png'
    cv2.imwrite(str1,img_resize)
    os.remove(input_file_name)
    count = count+1
