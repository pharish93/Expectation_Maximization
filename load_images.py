import os
import numpy as np
import cv2
import gc
import pandas as pd
import cPickle




def make_face_db(cache_file,image_size,start_index = 0 ,length = 100):

    image_resize_dimentions = image_size
    start_index = 0
    file_length = length
    # positive_face_base = './new/positive_face_'
    positive_face_base = './positive_faces/positive_face_'
    positive_label = 1
    negative_face_base = './negative_faces/negative_face_'
    negative_label = 0

    positive_face_db = load_images(positive_face_base, image_resize_dimentions,start_index, file_length, positive_label)
    negative_face_db = load_images(negative_face_base, image_resize_dimentions, start_index,file_length, negative_label)

    image_db = [positive_face_db, negative_face_db ]
    with open(cache_file, 'wb') as fid:
        cPickle.dump([image_db,image_size], fid, cPickle.HIGHEST_PROTOCOL)

    return image_db,image_size




def load_images(base_file_name, resize_dimentions, start_index, load_count,label):

    full_list = []
    for i in range(start_index, start_index+load_count):
        image_file_name = base_file_name + str(i) +'.png'
        if os.path.isfile(image_file_name):
            img = cv2.imread(image_file_name,0)
            img = cv2.resize(img,tuple(resize_dimentions))
            # img = np.array(img)
            full_list.append(list(img.flatten()))

    numpy_full_list = np.asarray(full_list)

    del full_list
    gc.collect()

    # face_db = pd.DataFrame(numpy_full_list)
    # face_db['label'] = label

    face_db = numpy_full_list
    return face_db