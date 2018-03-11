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


def expectations_t(image_db, parameters,pos):
    inv_std_face = inv(parameters['cov'][pos])
    exp_h = []
    exp_logh = []
    for element in image_db:
        diff = element - parameters['mu'][pos]
        diff = np.array(diff)[np.newaxis]
        e1 = np.matmul(diff, inv_std_face)
        e2 = np.matmul(e1, diff.T)

        a1 = (parameters['dof'][pos] + parameters['Dimensions']) / (parameters['dof'][pos] + e2.flatten())
        exp_h.append(a1)

        a2 = digamma(((parameters['dof'][pos] + parameters['Dimensions']) / 2)) - np.log(
            (parameters['dof'][pos] + e2.flatten()) / 2)

        exp_logh.append(a2)

    return [exp_h, exp_logh]


# assign every data point to its most likely cluster
def expectation(image_db, parameters):
    prob_mat = np.zeros((image_db.shape[0], parameters['num_tdist']))
    exp_h =  np.zeros((image_db.shape[0], parameters['num_tdist']))
    exp_logh =  np.zeros((image_db.shape[0], parameters['num_tdist']))

    for j in range(parameters['num_tdist']):
        p = parameters['lambda'][j]
        pdf = np.array(tdist_prob(image_db,parameters,j))
        pdf = pdf.flatten()
        prob_mat[:,j] = p * pdf
        temp_h, temp_logh = expectations_t(image_db,parameters,j)
        exp_h[:,j] = np.array(temp_h).flatten()
        exp_logh[:,j] = np.array(temp_logh).flatten()

    # normalize

    for i in range(image_db.shape[0]):
        sum_elem = np.sum(prob_mat[i,:])
        prob_mat[i,:] = prob_mat[i,:]/sum_elem


    return [prob_mat, exp_h,exp_logh]

def maximization(image_db,parameters, hidden_variable_estimates):

    pdf = hidden_variable_estimates[0]
    exp_h = hidden_variable_estimates[1]
    exp_logh = hidden_variable_estimates[2]
    for j in range(parameters['num_tdist']):
        parameters['lambda'][j] = np.sum(pdf[:,j])/np.sum(pdf)

        tp1 = np.array(pdf[:,j])[np.newaxis].T
        tp2 = np.array(exp_h[:,j])[np.newaxis].T
        tp3 = tp1*tp2*image_db
        parameters['mu'][j] = tp3.sum(axis=0)/np.sum(pdf[:,j]*exp_h[:,j])

        parameters['cov'][j] = np.zeros((image_db.shape[1],image_db.shape[1]))

        for i in range(image_db.shape[0]):
            diff = image_db[i, :] - parameters['mu'][j]
            diff = np.array(diff)[np.newaxis]
            parameters['cov'][j] += exp_h[i,j]* pdf[i,j]*np.matmul(diff.T, diff)

        parameters['cov'][j] = parameters['cov'][j]/np.sum(pdf[:,j]*exp_h[:,j])
        parameters['dof'][j] = fminbound(t_cost_calculation, 0, 999,
                                      args=(exp_h[:,j], exp_logh[:,j]))

    parameters['lambda'] = parameters['lambda'] / np.sum(parameters['lambda'])

    return parameters


def store_images(params, db_idx, image_size,count):
    str_file_name = './models/model4/model4_'

    if db_idx == 0:
        str_file_name += str(count)+'positive_face_'
    else:
        str_file_name += str(count)+'negative_face_'

    for i in range(params['num_tdist']):

        mu_face_img = params['mu'][i].reshape(image_size)
        str_mean = str_file_name + '_mean_'+ str(i) + '.png'
        cv2.imwrite(str_mean, mu_face_img)

        std_face = params['cov'][i]
        std_face_diga = np.sqrt(np.diag(std_face))
        std_face_img = (std_face_diga.reshape(image_size) - std_face_diga.min()) / abs(
            std_face_diga.max() - std_face_diga.min()) * 255

        str_std = str_file_name + '_std_' +str(i) + '.png'
        cv2.imwrite(str_std, std_face_img)

        print "Dof computed for the equations is :", params['dof'][i]


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

def tdist_prob(img_db, params,pos):
    mu_face = params['mu'][pos]
    std_face = params['cov'][pos]
    dof_face = params['dof'][pos]
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

    si = np.array(si).ravel()
    si = si*c

    return si



def model4(image_db, image_size,num_tdist):
    model_attributes = []
    # image_size = (image_size[0], image_size[1] , 3)

    for db_idx, loaded_image_db in enumerate(image_db):
        mu_modelling = np.random.rand( num_tdist, loaded_image_db.shape[1])*255
        std_modelling = np.zeros((num_tdist, loaded_image_db.shape[1],  loaded_image_db.shape[1]))
        for i in range(num_tdist):
            k = np.ones(loaded_image_db.shape[1])*4000
            std_modelling[i] = np.diag(k)

        lambda_modelling = np.random.rand(num_tdist)
        lambda_modelling = lambda_modelling / np.sum(lambda_modelling)
        v = 100*np.ones(num_tdist)
        D = loaded_image_db.shape[1]
        initial_guess = {
            'num_tdist': num_tdist,
            'lambda': lambda_modelling,
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
    plt.title('Model 4 ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def model4_test(test_image_db, model_attributes):
    # parameters = model_attributes
    # model_attributes = model_attributes[0]

    class_labeling = []
    misclassification = 0

    for image_db_idx, loaded_image_db in enumerate(test_image_db):

        measured_prob = []
        for idx, parameters in enumerate(model_attributes):
            prob_mat = np.zeros((loaded_image_db.shape[0],parameters['num_tdist']))

            for j in range(parameters['num_tdist']):
                p = parameters['lambda'][j]
                pdf = np.array(tdist_prob(loaded_image_db,parameters,j))
                pdf = pdf.flatten()

                prob_mat[:, j] = p * pdf

            temp = prob_mat.sum(axis=1)
            measured_prob.append(temp)

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
