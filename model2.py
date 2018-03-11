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
    pre_multiply = -0.5 * (log_det_std[1])
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


# assign every data point to its most likely cluster
def expectation(image_db, parameters):
    prob_mat = np.zeros((image_db.shape[0],parameters['num_gauss']))

    for j in range(parameters['num_gauss']):
        p = parameters['lambda'][j]
        model_attributes = [parameters['mu'][j], parameters['cov'][j] ]
        # pdf = np.array(log_multinomial_pdf(model_attributes,image_db))
        pdf = np.array(custom_multinomial_pdf(model_attributes,image_db))
        pdf = pdf.flatten()

        # pdf2 = multivariate_normal.pdf(image_db, mean= model_attributes[0], cov= model_attributes[1])
        # pdf = pdf2

        prob_mat[:,j] = p * pdf

    # normalize

    for i in range(image_db.shape[0]):
        sum_elem = np.sum(prob_mat[i,:])
        prob_mat[i,:] = prob_mat[i,:]/sum_elem

    return prob_mat


def maximization(image_db,parameters, pdf):

    for j in range(parameters['num_gauss']):
        parameters['lambda'][j] = np.sum(pdf[:,j])/np.sum(pdf)
        #
        # parameters['mu'][j] = np.zeros(image_db.shape[1])
        # for k in range(image_db.shape[0]):
        #     parameters['mu'][j] += pdf[k,j]*image_db[k,:]

        tp1 = np.array(pdf[:,j])[np.newaxis].T
        tp2 = tp1*image_db
        parameters['mu'][j] = tp2.sum(axis=0)/np.sum(pdf[:,j])

        parameters['cov'][j] = np.zeros((image_db.shape[1],image_db.shape[1]))

        for i in range(image_db.shape[0]):
            diff = image_db[i, :] - parameters['mu'][j]
            diff = np.array(diff)[np.newaxis]
            parameters['cov'][j] += pdf[i,j]*np.matmul(diff.T, diff)

        parameters['cov'][j] = parameters['cov'][j]/np.sum(pdf[:,j])

    parameters['lambda'] = parameters['lambda'] / np.sum(parameters['lambda'])

    return parameters


def store_images(params,db_idx,image_size):

    str_file_name = './models/model2/model2_'

    if db_idx == 0:
        str_file_name += 'positive_face_'
    else:
        str_file_name += 'negative_face_'

    for i in range(params['num_gauss']):

        mu_face_img = params['mu'][i].reshape(image_size)
        str_mean = str_file_name + '_mean_' + str(i)+'.png'
        cv2.imwrite(str_mean, mu_face_img)

        std_face = params['cov'][i]
        std_face_diga = np.sqrt(np.diag(std_face))
        std_face_img = (std_face_diga.reshape(image_size) - std_face_diga.min()) / abs(std_face_diga.max() - std_face_diga.min())*255

        str_std = str_file_name + '_std_' + str(i)+'.png'
        cv2.imwrite(str_std, std_face_img)





def model2(image_db,num_gauss,image_size):

    model_attributes = []

    for db_idx, loaded_image_db in enumerate(image_db) :
        mu_modelling = np.random.rand( num_gauss, loaded_image_db.shape[1])*255
        # std_modelling = np.random.rand(num_gauss, loaded_image_db.shape[1],  image_db[0].shape[1])*10
        std_modelling = np.zeros((num_gauss, loaded_image_db.shape[1],  loaded_image_db.shape[1]))

        for i in range(num_gauss):
            # k = np.random.rand(loaded_image_db.shape[1])*250
            k = np.ones(loaded_image_db.shape[1])*4000
            std_modelling[i] = np.diag(k)


        lambda_modelling = np.random.rand(num_gauss)
        lambda_modelling = lambda_modelling / np.sum(lambda_modelling)

        initial_guess = {
            'num_gauss': num_gauss,
            'mu': mu_modelling,
            'cov': std_modelling,
            'lambda' : lambda_modelling}

        params = initial_guess

        count = 0
        # while shift > epsilon:
        while count < 10:

            count += 1
            # E-step
            updated_labels = expectation(loaded_image_db, params)
            # M-step
            params = maximization(loaded_image_db,params,updated_labels)

        model_attributes.append(params)
        store_images(params,db_idx,image_size)

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
    plt.title('Model 2 ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def model2_test(test_image_db,model_attributes):

    # parameters = model_attributes
    misclassification = 0
    class_labeling = []

    for image_db_idx, loaded_image_db in enumerate(test_image_db):

        measured_prob = []
        for idx,parameters in enumerate(model_attributes):
            prob_mat = np.zeros((loaded_image_db.shape[0], parameters['num_gauss']))

            for j in range(parameters['num_gauss']):
                p = parameters['lambda'][j]
                one_gauss_model = [parameters['mu'][j], parameters['cov'][j]]
                pdf = np.array(custom_multinomial_pdf(one_gauss_model, loaded_image_db))
                pdf = pdf.flatten()

                prob_mat[:, j] = p * pdf

            temp = prob_mat.sum(axis=1)
            measured_prob.append(temp)


        count_positive = 0
        count_negative = 0
        predicted_lable = 10

        if (image_db_idx == 0):
            label = 1
        else:
            label = 0
        for i in range(len(measured_prob[0])):
            probability_val = (measured_prob[0][i] / (measured_prob[0][i] + measured_prob[1][i]))
            if probability_val > 0.5:
                predicted_lable = 1
                count_positive += 1
            else:
                predicted_lable = 0
                count_negative += 1
            class_labeling.append(
                [label, predicted_lable, probability_val, measured_prob[0][i], measured_prob[1][i]])

        print "count positive : ", count_positive
        print "count negative :", count_negative

        numerator = 0
        if image_db_idx == 0:
            numerator = count_positive
            flase_negatives = float(count_negative) / loaded_image_db.shape[0]
            print "False Negatives :", flase_negatives

            misclassification += count_negative
        if image_db_idx == 1:
            numerator = count_negative
            flase_positives = float(count_positive) / loaded_image_db.shape[0]
            print "False positive :", flase_positives
            misclassification += count_positive

        percent_correct = float(numerator) / (count_positive + count_negative) * 100
        print "testing for", image_db_idx, "is :", percent_correct

    class_labeling = np.array(class_labeling)
    orginal_lables = class_labeling[:, 0]
    prob_vals = class_labeling[:, 2]
    draw_roc_curve(orginal_lables, prob_vals)
    print "Misclassification Rate :", (float)(misclassification) / (test_image_db[0].shape[0] + test_image_db[1].shape[0])
