import cv2
import numpy as np
from scipy.special import digamma
from numpy.linalg import inv
from scipy.special import gammaln
from scipy.optimize import fminbound
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt


def t_cost_calculation(dof, e_h, e_logh):
    # e_h = hidden_variable_estimates[0]
    # e_logh = hidden_variable_estimates[1]
    half_dof = dof / 2
    I = len(e_h)
    t1 = half_dof * np.log(half_dof)
    t2 = gammaln(half_dof)
    finalCost = 0

    for i in range(I):
        t3 = (half_dof - 1) * e_logh[i]
        t4 = half_dof * e_h[i]
        finalCost = finalCost + t1 - t2 + t3 - t4

    finalCost = -finalCost

    return finalCost


# assign every data point to its most likely cluster
def expectation(image_db, parameters):

    inv_std_face = inv(parameters['cov'])
    exp_h = []
    exp_logh = []
    for element in image_db:
        diff = element - parameters['mu']
        diff = np.array(diff)[np.newaxis]
        e1 = np.matmul(diff, inv_std_face)
        e2 = np.matmul(e1, diff.T)

        a1 = (parameters['dof'] + parameters['Dimensions']) / (parameters['dof'] + e2.flatten())
        exp_h.append(a1)

        a2 = digamma(((parameters['dof'] + parameters['Dimensions']) / 2)) - np.log(
            (parameters['dof'] + e2.flatten()) / 2)

        exp_logh.append(a2)

    return [exp_h, exp_logh]


def maximization(image_db, parameters, hidden_variable_estimates):

    exp_h = hidden_variable_estimates[0]
    tp2 = exp_h * image_db
    parameters['mu'] = tp2.sum(axis=0) / np.sum(exp_h)

    parameters['cov'] = np.zeros((image_db.shape[1], image_db.shape[1]))

    for i in range(image_db.shape[0]):
        diff = image_db[i, :] - parameters['mu']
        diff = np.array(diff)[np.newaxis]
        parameters['cov'] += exp_h[i] * np.matmul(diff.T, diff) / np.sum(exp_h)

    parameters['dof'] = fminbound(t_cost_calculation, 0, 999,
                                  args=(hidden_variable_estimates[0], hidden_variable_estimates[1]))

    return parameters


def store_images(params, db_idx, image_size,count):
    str_file_name = './models/model3/model3_'

    if db_idx == 0:
        str_file_name += str(count)+'positive_face_'
    else:
        str_file_name += str(count)+'negative_face_'

    mu_face_img = params['mu'].reshape(image_size)
    str_mean = str_file_name + '_mean' + '.png'
    cv2.imwrite(str_mean, mu_face_img)

    std_face = params['cov']
    std_face_diga = np.sqrt(np.diag(std_face))
    std_face_img = (std_face_diga.reshape(image_size) - std_face_diga.min()) / abs(
        std_face_diga.max() - std_face_diga.min()) * 255

    str_std = str_file_name + '_std_' + '.png'
    cv2.imwrite(str_std, std_face_img)

    print "Dof computed for the equations is :", params['dof']


def compute_likelihood(img_db, params):
    mu_face = params['mu']
    std_face = params['cov']
    dof_face = params['dof']
    dim_face = params['Dimensions']

    inv_std_face = inv(std_face)
    si = []
    sum_up = 0

    for element in img_db:
        diff = element - mu_face
        diff = np.array(diff)[np.newaxis]
        e1 = np.matmul(diff, inv_std_face)
        e2 = np.matmul(e1, diff.T)
        si.append(e2)
        sum_up += np.log(1+e2/dof_face)/2

    t1 = gammaln((dof_face + dim_face) / 2)
    t2 = dim_face*np.log(dof_face*np.pi)/2
    t3 = np.linalg.slogdet(std_face)[1]/2
    t4 = gammaln(dof_face/2)

    L1 = (t1 - t2 - t3 - t4)*img_db.shape[0]

    L = L1 - (dof_face + dim_face)*sum_up

    return L

def log_likelihood(img_db, params):
    mu_face = params['mu']
    std_face = params['cov']
    dof_face = params['dof']
    dim_face = params['Dimensions']

    inv_std_face = inv(std_face)
    si = []

    t1 = gammaln((dof_face + dim_face) / 2)
    t2 = dim_face*np.log(dof_face*np.pi)/2
    t3 = np.linalg.slogdet(std_face)[1]/2
    t4 = gammaln(dof_face/2)

    fixed_pre_term = (t1 - t2 - t3 - t4)

    for element in img_db:
        diff = element - mu_face
        diff = np.array(diff)[np.newaxis]
        e1 = np.matmul(diff, inv_std_face)
        e2 = np.matmul(e1, diff.T)
        e3 = np.log(1+e2/dof_face)
        e4 = fixed_pre_term - ((dof_face + dim_face)/2 * e3.flatten())
        si.append(e4)

    return si

def tdist_prob(img_db, params):
    mu_face = params['mu']
    std_face = params['cov']
    dof_face = params['dof']
    dim_face = params['Dimensions']

    inv_std_face = inv(std_face)
    si = []

    t1 = gammaln((dof_face + dim_face) / 2)
    # t2 = dim_face*np.log(dof_face*np.pi)/2
    # t3 = np.linalg.slogdet(std_face)[1]/2
    t4 = gammaln(dof_face/2)
    c =  np.exp(t1-t4)
    c = c / (pow((dof_face * np.pi),(dim_face / 2)) * np.sqrt(np.linalg.det(std_face)))

    # fixed_pre_term = (t1 - t2 - t3 - t4)

    for element in img_db:
        diff = element - mu_face
        diff = np.array(diff)[np.newaxis]
        e1 = np.matmul(diff, inv_std_face)
        e2 = np.matmul(e1, diff.T)
        e3 = (1+e2/dof_face)
        e4 = pow(e3, -1 * (dof_face + dim_face) / 2)
        # e4 = fixed_pre_term - ((dof_face + dim_face)/2 * e3.flatten())
        si.append(e4)

    si = si*c

    return si



def model3(image_db, image_size):
    model_attributes = []
    # image_size = (image_size[0], image_size[1] , 3)

    for db_idx, loaded_image_db in enumerate(image_db):
        mu_modelling = np.random.rand(loaded_image_db.shape[1]) * 255
        k = np.ones(loaded_image_db.shape[1]) * 4000
        std_modelling = np.diag(k)

        v = 100
        D = loaded_image_db.shape[1]
        initial_guess = {
            'dof': v,
            'Dimensions': D,
            'mu': mu_modelling,
            'cov': std_modelling
        }

        params = initial_guess

        count = 0
        # while shift > epsilon:
        while count < 15:
            count += 1
            # E-step
            hidden_estimates = expectation(loaded_image_db, params)
            # M-step
            params = maximization(loaded_image_db, params, hidden_estimates)
            # L = compute_likelihood(params,loaded_image_db)
            print count , params['dof']
            store_images(params, db_idx, image_size,count)

            a = 1

        model_attributes.append(params)
        store_images(params, db_idx, image_size,count)


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
    plt.title('Model 3 ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def model3_test(test_image_db, model_attributes):
    # parameters = model_attributes
    # model_attributes = model_attributes[0]

    class_labeling = []
    misclassification = 0

    for image_db_idx, loaded_image_db in enumerate(test_image_db):

        measured_prob = []
        for idx, parameters in enumerate(model_attributes):
            # prob_mat = np.zeros((loaded_image_db.shape[0]))

            # log_prob_mat = log_likelihood(loaded_image_db,parameters)
            prob_mat = tdist_prob(loaded_image_db,parameters)
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
    orginal_lables = class_labeling[:,0]
    prob_vals =  class_labeling[:,2]
    draw_roc_curve(orginal_lables,prob_vals)
    print "Misclassification Rate :", misclassification / (test_image_db[0].shape[0] + test_image_db[1].shape[0])
