import cPickle
import os
from load_images import *
from model1 import *
from model2 import *
from model3 import *
from model4 import *
from modle5 import *

def main():

    train_cache_file = './cache/train_face_data.pkl'
    test_cache_file = './cache/test_face_data.pkl'

    image_size = (5,5)
    if os.path.exists(train_cache_file):
        with open(train_cache_file, 'rb') as fid:
            train_image_db,image_size = cPickle.load(fid)
        with open(test_cache_file, 'rb') as fid:
            test_image_db,image_size = cPickle.load(fid)
    else :
        train_image_db,image_size = make_face_db(train_cache_file, image_size, 0 ,1000)
        test_image_db, image_size = make_face_db(test_cache_file, image_size,1000,200)

    # MODEL 1 -- 30,30,3
    # if os.path.exists('./cache/model1.pkl'):
    #     with open('./cache/model1.pkl', 'rb') as fid:
    #         model1_attributes = cPickle.load(fid)
    # else:
    #     model1_attributes = model1(train_image_db,image_size)
    #     with open('./cache/model1.pkl', 'wb') as fid:
    #         cPickle.dump(model1_attributes, fid, cPickle.HIGHEST_PROTOCOL)
    # model1_test(test_image_db,model1_attributes)

    # MODEL 2

    # num_gauss = 5
    # if os.path.exists('./cache/model2.pkl'):
    #     with open('./cache/model2.pkl', 'rb') as fid:
    #         model2_attributes = cPickle.load(fid)
    # else:
    #     model2_attributes = model2(train_image_db,num_gauss,image_size)
    #     with open('./cache/model2.pkl', 'wb') as fid:
    #         cPickle.dump(model2_attributes, fid, cPickle.HIGHEST_PROTOCOL)
    #
    # model2_test(test_image_db,model2_attributes)


    # Model 3 -- 30,30
    # if os.path.exists('./cache/model3.pkl'):
    #     with open('./cache/model3.pkl', 'rb') as fid:
    #         model3_attributes = cPickle.load(fid)
    # else:
    #     model3_attributes = model3(train_image_db,image_size)
    #     with open('./cache/model3.pkl', 'wb') as fid:
    #         cPickle.dump(model3_attributes, fid, cPickle.HIGHEST_PROTOCOL)
    # model3_test(test_image_db,model3_attributes)



    # Model 4 -- 30,30
    num_tdist = 5
    if os.path.exists('./cache/model4.pkl'):
        with open('./cache/model4.pkl', 'rb') as fid:
            model4_attributes = cPickle.load(fid)
    else:
        model4_attributes = model4(train_image_db,image_size,num_tdist)
        with open('./cache/model4.pkl', 'wb') as fid:
            cPickle.dump(model4_attributes, fid, cPickle.HIGHEST_PROTOCOL)
    model4_test(test_image_db,model4_attributes)


    #
    # # # MODEL 5 - 30 30 3
    # num_factors = 5
    # if os.path.exists('./cache/model5.pkl'):
    #     with open('./cache/model5.pkl', 'rb') as fid:
    #         model5_attributes = cPickle.load(fid)
    # else:
    #     model5_attributes = model5(train_image_db,num_factors,image_size)
    #     with open('./cache/model5.pkl', 'wb') as fid:
    #         cPickle.dump(model5_attributes, fid, cPickle.HIGHEST_PROTOCOL)
    #
    # model5_test(test_image_db,model5_attributes)

    a = 1

if __name__ == '__main__':
    main()
