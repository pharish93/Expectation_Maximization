import cv2
import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import math
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

def log_multinomial_pdf(model_attributes, img_db):
    mu_face = model_attributes[0]
    std_face = model_attributes[1]

    inv_std_face = inv(std_face)
    log_det_std = np.linalg.slogdet(std_face)
    temp = -0.5*img_db.shape[1]*np.log(2*np.pi)
    pre_multiply = (-0.5 * (log_det_std[1]))
    pre_multiply = pre_multiply + temp
    pdf = []

    for element in img_db:
        diff = element - mu_face
        diff = np.array(diff)[np.newaxis]
        e1 = np.matmul(diff, inv_std_face)
        e2 = np.matmul(e1, diff.T)
        exp_part_exp = (-0.5 * e2)
        exp_part_exp = (exp_part_exp.flatten())
        k1 = pre_multiply + exp_part_exp
        pdf.append(k1)

    return pdf

def custom_multinomial_pdf(model_attributes, img_db):
    mu_face = model_attributes[0]
    std_face = model_attributes[1]

    inv_std_face = inv(std_face)
    pre_multiply = np.linalg.det(std_face)
    pre_multiply = 1 / np.sqrt(pre_multiply)
    pdf = []

    for element in img_db:
        diff = element - mu_face
        diff = np.array(diff)[np.newaxis]
        e1 = np.matmul(diff, inv_std_face)
        e2 = np.matmul(e1, diff.T)
        exp_part_exp = np.exp(-0.5 * e2)
        exp_part_exp = (exp_part_exp.flatten())
        k1 = pre_multiply* exp_part_exp
        pdf.append(k1)

    return pdf

def model1(image_db, image_size):
    # image_size = (image_size[0],image_size[1],3)
    model_attributes = []

    count = 0

    for loaded_image_db in image_db :
        mu_face = np.zeros(loaded_image_db.shape[1])
        std_face = np.zeros((loaded_image_db.shape[1],loaded_image_db.shape[1]))

        for i in range(loaded_image_db.shape[0]):
            mu_face += loaded_image_db[i, :]

        mu_face /= loaded_image_db.shape[0]

        for i in range(loaded_image_db.shape[0]):
            diff = loaded_image_db[i,:] - mu_face
            diff = np.array(diff)[np.newaxis]
            std_face += np.matmul(diff.T,diff)
        std_face /= loaded_image_db.shape[0]

        # k = (std_face.T == std_face).all()
        mu_face_img = mu_face.reshape(image_size)

        # std_face = np.diag(np.diag(std_face))
        std_face_diga = np.sqrt(np.diag(std_face))
        std_face_img = (std_face_diga.reshape(image_size) - std_face_diga.min()) / abs(std_face_diga.max() - std_face_diga.min())*255


        model_attributes.append(([mu_face,std_face]))
        str_mean = './models/model1/model1_mean_face_image_' + str(count) + '.png'
        cv2.imwrite(str_mean, mu_face_img)
        str_std = './models/model1/model1_std_face_image_' + str(count) + '.png'
        cv2.imwrite(str_std, std_face_img)
        count += 1


    return model_attributes


def draw_roc_curve(orginal_lables,prob_vals):

    false_positive_rate, true_positive_rate, _ = roc_curve(orginal_lables.ravel(), prob_vals.ravel())
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure()
    lw = 2
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model 1 ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def model1_test(test_image_db,model_attributes):

    class_labeling = []
    misclassification = 0
    prob = []
    for i in range(len(test_image_db)):
        # pdf_positive = log_multinomial_pdf(model_attributes[0],test_image_db[i])
        # pdf_negative = log_multinomial_pdf(model_attributes[1],test_image_db[i])

        pdf_positive = custom_multinomial_pdf(model_attributes[0],test_image_db[i])
        pdf_negative = custom_multinomial_pdf(model_attributes[1],test_image_db[i])

        count_positive = 0
        count_negative = 0

        predicted_lable = -10

        if(i==0):label = 1
        else: label = 0
        for j in range(len(pdf_positive)):
            probability_val = (pdf_positive[j]/(pdf_positive[j]+pdf_negative[j]))
            if probability_val >= 0.5:
                predicted_lable = 1
                count_positive += 1
            else:
                predicted_lable = 0
                count_negative += 1
            class_labeling.append( [label, predicted_lable, probability_val, pdf_positive[j], pdf_negative[j]])

        print "count positive : ", count_positive
        print "count negative :", count_negative

        numerator = 0
        if i == 0:
            numerator = count_positive
            flase_negatives = float(count_negative)/len(pdf_positive)
            print "False Negatives :", flase_negatives

            misclassification += count_negative
        if i == 1:
            numerator = count_negative
            flase_positives = float(count_positive)/len(pdf_positive)
            print "False positive :", flase_positives

            misclassification += count_positive

        percent_correct = float(numerator)/(count_positive+count_negative)*100
        print "testing for", i , "is :", percent_correct


    class_labeling = np.array(class_labeling)

    orginal_lables = class_labeling[:,0]
    prob_vals =  class_labeling[:,2]
    draw_roc_curve(orginal_lables,prob_vals)
    print "Misclassification Rate :", float(misclassification)/(test_image_db[0].shape[0] + test_image_db[1].shape[0])
