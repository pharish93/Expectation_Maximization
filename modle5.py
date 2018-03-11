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
        k1 = pre_multiply * exp_part_exp
        pdf.append(k1)

    return pdf


# assign every data point to its most likely cluster
def expectation(image_db, parameters):

    inv_std_face = inv(parameters['cov'])
    phi = parameters['phi']

    t1 = np.matmul(phi.T,inv_std_face)
    t2 = inv(np.matmul(t1,phi) + np.identity(parameters['num_factors']))
    t3 = np.matmul(t2,t1)

    E_hi = []
    E_hi_hitrans = []

    for i in range(image_db.shape[0]):
        diff = image_db[i, :] - parameters['mu']
        diff = np.array(diff)[np.newaxis]
        temp = np.matmul(t3,diff.T)
        E_hi.append(temp)
        E_hi_hitrans.append(t2 + np.matmul(temp,temp.T))

    return [E_hi, E_hi_hitrans]


def maximization(image_db, parameters, hidden_estimates):

    E_hi = hidden_estimates[0]
    E_hi_hitrans = hidden_estimates[1]

    phi_temp1 = np.zeros((image_db.shape[1],parameters['num_factors']))
    phi_temp2 = np.zeros((parameters['num_factors'],parameters['num_factors']))

    for i in range(image_db.shape[0]):
        diff = image_db[i, :] - parameters['mu']
        diff = np.array(diff)[np.newaxis]

        phi_temp1 += np.matmul(diff.T,E_hi[i].T)
        phi_temp2 += E_hi_hitrans[i]

    phi_temp2 = inv(phi_temp2)
    phi = np.matmul(phi_temp1,phi_temp2)
    parameters['phi'] = phi

    cov_diag = np.zeros((image_db.shape[1],1))
    for i in range(image_db.shape[0]):
        diff = image_db[i, :] - parameters['mu']
        diff = np.array(diff)[np.newaxis]

        t1 = diff.T*diff.T
        t2 = np.matmul(phi,E_hi[i])
        t3 = t2*diff.T

        cov_diag += t1 - t3

    cov_diag/= image_db.shape[0]
    cov_diag = np.diag(cov_diag.flatten())
    parameters['cov'] = cov_diag

    return parameters



def store_images(params, db_idx, image_size):
    str_file_name = './models/model5/model5_'

    if db_idx == 0:
        str_file_name += 'positive_face_'
    else:
        str_file_name += 'negative_face_'

    mu_face_img = params['mu'].reshape(image_size)
    str_mean = str_file_name + '_mean' + '.png'
    cv2.imwrite(str_mean, mu_face_img)

    std_face = params['cov']
    std_face_diga = np.sqrt(np.diag(std_face))
    std_face_img = (std_face_diga.reshape(image_size) - std_face_diga.min()) / abs(
        std_face_diga.max() - std_face_diga.min()) * 255

    str_std = str_file_name + '_std' + '.png'
    cv2.imwrite(str_std, std_face_img)

    c = 2
    for i in range(params['num_factors']):
        temp_face = params['mu'] - c* params['phi'][:,i]
        mu_face_img = temp_face.reshape(image_size)
        str_mean = str_file_name + '_factor_analysis_negative_' + str(i)+ '.png'
        cv2.imwrite(str_mean, mu_face_img)

        temp_face = params['mu'] + c* params['phi'][:,i]
        mu_face_img = temp_face.reshape(image_size)
        str_mean = str_file_name + '_factor_analysis_positive_' + str(i)+ '.png'
        cv2.imwrite(str_mean, mu_face_img)

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
    plt.title('Model 5 ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def model5(image_db, num_factors, image_size):
    model_attributes = []
    image_size = (image_size[0], image_size[1] , 3)
    for db_idx, loaded_image_db in enumerate(image_db):

        mu_modelling = np.zeros((loaded_image_db.shape[1]))
        std_modelling = np.zeros((loaded_image_db.shape[1], loaded_image_db.shape[1]))

        for i in range(loaded_image_db.shape[0]):
            mu_modelling += loaded_image_db[i, :]

        mu_modelling /= loaded_image_db.shape[0]

        for i in range(loaded_image_db.shape[0]):
            diff = loaded_image_db[i, :] - mu_modelling
            diff = np.array(diff)[np.newaxis]
            std_modelling += np.matmul(diff.T, diff)
        std_modelling /= loaded_image_db.shape[0]
        std_modelling = np.diag(np.diag(std_modelling))

        phi_modelling = np.random.rand(loaded_image_db.shape[1],num_factors)

        initial_guess = {
            'num_factors': num_factors,
            'mu': mu_modelling,
            'cov': std_modelling,
            'phi': phi_modelling}

        params = initial_guess

        count = 0
        # while shift > epsilon:
        while count < 10:
            count += 1
            # E-step
            hidden_estimates = expectation(loaded_image_db, params)
            # M-step
            params = maximization(loaded_image_db, params, hidden_estimates)

        model_attributes.append(params)
        store_images(params, db_idx, image_size)

    return model_attributes


def model5_test(test_image_db, model_attributes):
    # model_attributes = model_attributes[0]
    misclassification = 0
    class_labeling = []

    for image_db_idx, loaded_image_db in enumerate(test_image_db):

        measured_prob = []

        for idx, parameters in enumerate(model_attributes):
            # prob_mat = np.zeros((loaded_image_db.shape[0]))

            cov = parameters['cov'] + np.matmul(parameters['phi'],parameters['phi'].T)
            one_gauss_model = [parameters['mu'], cov]

            # prob_mat = log_multinomial_pdf(one_gauss_model, loaded_image_db)
            prob_mat = custom_multinomial_pdf(one_gauss_model, loaded_image_db)
            measured_prob.append(prob_mat)

        count_positive = 0
        count_negative = 0
        predicted_lable = 10

        if(image_db_idx==0):label = 1
        else: label = 0
        for i in range(len(measured_prob[0])):
            probability_val = (measured_prob[0][i] / (measured_prob[0][i] + measured_prob[1][i]))
            if probability_val > 0.5:
                predicted_lable = 1
                count_positive += 1
            else:
                predicted_lable = 0
                count_negative += 1
            class_labeling.append( [label, predicted_lable, probability_val, measured_prob[0][i],  measured_prob[1][i]])

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
    print "Misclassification Rate :", misclassification / (test_image_db[0].shape[0] + test_image_db[1].shape[0])
    a = 1
