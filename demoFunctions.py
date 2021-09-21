# -*- coding: utf-8 -*-

# import libraries

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers 
from keras import Sequential
from keras.losses import categorical_crossentropy as CCE

from sklearn.model_selection import train_test_split
import networkx as nx
import time

from multiprocessing import Process, Manager, Value, Array
from multiprocessing.pool import ThreadPool as Pool


def dataset_preparation(x_train, x_test, y_train, y_test):

    # prepare the images: normalization, padding (from 28x28 to 32x32), train-test split, one-hot encoding of the labels

    # Rescale the images from [0,255] to the [0.0,1.0] range
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    # Pad the images to shape 32x32
    x_train, x_test = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant'), np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
    classes_OH = np.eye(10)
    y_OH = classes_OH[y_test]
    y_train_OH = classes_OH[y_train]

    return x_train, y_train_OH, x_test, y_OH


def LeNet5(inputShape=(32,32,1)):
    
    # define CNN model to be attacked

    model = keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=5, activation='relu',
              input_shape=inputShape)) # 28x28x6
    model.add(layers.AveragePooling2D()) # 14x14x6

    model.add(layers.Conv2D(filters=16, kernel_size=5, activation='relu')) # 10x10x16
    model.add(layers.AveragePooling2D()) # 5x5x16

    model.add(layers.Flatten()) # 400

    model.add(layers.Dense(units=120, activation='relu'))

    model.add(layers.Dense(units=84, activation='relu'))

    model.add(layers.Dense(units=10, activation = 'softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    print(model.summary(), flush=True)

    return model


def pick_balanced_labels(xT, yT_OH, model, images_per_class):

    # create the balanced dataset xT of (images_per_classes * n. of classes) images

    # create the ordered list of labels (with their respective indexes)
    ordered_labels = []
    preds = model.predict(xT)

    for i in range(len(yT_OH[0])): # number of classes
        labels_i = []
        for k in range(xT.shape[0]): # images in the test set
            # if the image belongs to class k and is correctly classified by the model
            if np.argmax(yT_OH[k])==i and (yT_OH[k].argmax() == preds[k].argmax()):
                labels_i.append(k)
        ordered_labels.append(labels_i)
    
    chosen_indexes = []
    for i in range(len(yT_OH[0])):
        np.random.seed(1)
        chosen_indexes.append(np.random.choice(ordered_labels[i], images_per_class, replace = False))

    return chosen_indexes # indexes of the n_classes*images_per_class selected pictures


####################################### Zeroth Order Inexact Update #######################################

def bound_image(pic):

    # keep the pixels' value between 0 and 1
    
    # pic[pic<0] = 0
    # pic[pic>1] = 1

    return pic


def stochastic_gradient(img_t, label_OH, model, c_t, m):

    # img_t: 3D array, input image
    # label_OH: 1D array, one hot encoded real label of the instance image
    # c_t: float, smoothing parameter
    # m: integer, number of sampled normals at each step

    # image loss at step t
    loss_t = tf.cast(CCE(label_OH, model.predict(img_t[np.newaxis, ...]).squeeze()), dtype=tf.float64)

    # m random perturbation directions
    m_directions = np.random.standard_normal((m, *img_t.shape)) #(m,32,32,1)
    
    # add perturbation along the gaussian direction to the image at step t
    m_imgs_t = img_t + m_directions*c_t
    
    # keep the pixels' value between 0 and 1
    m_imgs_t = bound_image(m_imgs_t)
    
    # Compute the loss at step t on the perturbed image
    m_loss_perturbed_t = tf.cast(CCE(np.repeat(label_OH[np.newaxis,...], m, axis=0), model.predict(m_imgs_t)), dtype=tf.float64)

    # compute the gradient
    sum_term = (m_loss_perturbed_t - loss_t)/c_t

    # initialize gradients for each random direction
    grad_dir_t = np.empty((m,*img_t.shape))
    for term in range(m):
        grad_dir_t[term] = sum_term[term]*m_directions[term]

    # average gradient
    grad_t = np.sum(grad_dir_t, axis=0).squeeze() / m 

    return grad_t


def ICG(x, g, gamma, mu, xi, L_norm, z0):

    # x: 3D array, input image
    # g: 2D array, gradient
    # gamma: float, ICG momentum constant
    # mu: float, exit condition for the ICG
    # xi: float, bound of the perturbation
    # L_norm: string, norm type
    # z0: 3D array, original image

    # maximum number of ICG iterations
    maxItICG = 1000
    x = x.squeeze()
    # initialize perturbed image
    y_mean = x.copy()
    # initialize exit condition flag
    flag = 0
    # iteration number
    Iter = 1
    z0 = z0.squeeze()
    
    while flag == 0:
        
        # gradient computation step
        first_term = g + gamma*(y_mean - x)
        if L_norm == 'L_inf':
            y_iter = z0 - xi * np.sign(first_term)
        elif L_norm == 'L_2':
            pass
        elif L_norm == 'L_1':
            pass
        else:
            print('Invalid L-norm string', flush=True)
            return
        
        # compute h(y_iter)
        h = np.dot(first_term.flatten(), (y_iter - y_mean).flatten())

        # check exit condition
        if (Iter >= maxItICG) or (h >= -mu):
            flag = 1
        else:
            y_mean = ((Iter-1)/(Iter+1))*y_mean+(2/(Iter+1))*y_iter
            Iter += 1

        # keep the pixels' value between 0 and 1
        y_mean = bound_image(y_mean)

    print(f'ICG completed after {Iter} iterations.', flush=True)

    return y_mean[...,np.newaxis]


def GM_IU(z_t, label_OH, y, model, d, L_norm, convex, xi, maxIt, mu, c_t, gamma_t, m):

    # Zeroth Order Gradient Method with Inexact Update

    # z_t: 3D array, perturbed image at each step t 
    # label_OH: 1D array, one hot encoded real label of the instance image
    # y: 1D array, predicted label of the non-perturbed image
    # d: integer, dimensionality of the problem
    # L_norm: string, norm type
    # convex: boolean, flag to decide whether apply algorithm in the convex case
    # L: float, upperbound for the Lipschitz constant
    # xi: float, bound of the perturbation
    # maxIt: integer, maximum number of iterations
    # mu: float, exit condition for the ICG
    # c_t: float, smoothing parameter
    # gamma_t: float, ICG momentum constant
    # m: integer, number of sampled normals at each step

    # x_t: perturbed image up to step t
    # y_new: label of the image computed at step t
    

    # input image
    z0 = z_t.copy()
    
    # step image
    x_t = z_t.copy()
    
    # gradient initialization
    grad_t = np.empty((z_t.shape[0], z_t.shape[1]))

    # check if image is already misclassified
    isOriginallyMiscl = np.argmax(label_OH) != np.argmax(y)
    
    if isOriginallyMiscl:
        print('The image is already misclassified by the model: exiting', flush=True)
        return x_t, -1
    
    # START ALGORITHM
    for t in range(maxIt): # iteration number

        print('Iterazione n.', t+1, flush=True)

        if convex:

            # initialize momentum
            alpha_t = 2/(t+2)
            # convex combination before the gradient estimate
            w_t = (1-alpha_t)*z_t + alpha_t*x_t
            # estimate gradient
            grad_t = stochastic_gradient(w_t, label_OH, model, c_t, m)

        else:
            # estimate gradient
            grad_t = stochastic_gradient(x_t, label_OH, model, c_t, m)
        
        # compute ICG
        x_t = ICG(x_t, grad_t, gamma_t, mu, xi, L_norm, z0)

        if convex:
            # update z_t
            z_t = (1-alpha_t)*z_t + alpha_t*x_t

            new_label = model.predict(z_t[np.newaxis,...]).argmax()
            if new_label!=label_OH.argmax():
                print(f'\nNew label: {new_label}\n', flush=True)
                f, axis = plt.subplots(1,3, figsize=(15,5))
                print('Original Image - Perturbed Image - Noise ', flush=True)
                axis[0].imshow(z0.squeeze(), vmin = 0, vmax = 1)
                axis[1].imshow((z_t).squeeze(), vmin = 0, vmax = 1)
                axis[2].imshow((z_t-z0).squeeze(), vmin = 0, vmax = 1)
                plt.show()
                return z_t, new_label

        else:
            # keep the pixels' value between 0 and 1
            x_t = bound_image(x_t)
            
            new_label = model.predict(x_t[np.newaxis,...]).argmax()
            if new_label != label_OH.argmax():
                print(f'\nNew label: {new_label}\n', flush=True)
                f, axis = plt.subplots(1,3, figsize=(15,5))
                print('Original Image - Perturbed Image - Noise ', flush=True)
                axis[0].imshow(z0.squeeze(), vmin = 0, vmax = 1)
                axis[1].imshow((x_t).squeeze(), vmin = 0, vmax = 1)
                axis[2].imshow((x_t-z0).squeeze(), vmin = 0, vmax = 1)
                plt.show()
                return x_t, new_label

    print('\nImage not misclassified.\n\n', flush=True)

    return  x_t, -1


def setup_IU(xT, y_OH, model, instances_per_class=100):

    # selection of instances_per_class * (number of classes) images to be considered

    # xT: 4D array, images dataset (MNIST test set in our case)
    # y_OH: 2D array, one hot encoded labels of the images dataset
    # instances_per_class: integer, number of picked images per class
    # chosen_indexes: list, indexes of the chosen images
    # x_t_chosen: 4D array, chosen images dataset
    # label_OH_chosen: 2D array, one hot encoded labels of the chosen images dataset
    
    # choose the indexes
    chosen_indexes = pick_balanced_labels(xT, y_OH, model, instances_per_class)

    # pick just instances_per_class * (number of classes) images (they are ordered with respect to the labels)
    x_t_chosen = np.empty((len(chosen_indexes)*len(chosen_indexes[0]), *(xT[0].shape)))
    label_OH_chosen = np.empty((len(chosen_indexes)*len(chosen_indexes[0]), len(y_OH[0])))
    for i in range(len(chosen_indexes)):
        for j in range(len(chosen_indexes[i])):
            x_t_chosen[i*len(chosen_indexes[0]) + j] = xT[chosen_indexes[i][j]]
            label_OH_chosen[i*len(chosen_indexes[0]) + j] = y_OH[chosen_indexes[i][j]]
            
    return chosen_indexes, x_t_chosen, label_OH_chosen


def main_IU(model, xT, y_OH):

    # run Zeroth Order Inexact Update

    # model: keras.engine.sequential.Sequential or keras.engine.training.Model, target model

    # initialize data
    instances_per_class = 100
    chosen_indexes, x_t_chosen, label_OH_chosen = setup_IU(xT, y_OH, model, instances_per_class)
    # model output on the original selected images
    y = model.predict(x_t_chosen)

    d = 1024
    maxIt = 100
    xi = 0.4
    L_norm = 'L_inf'
    convex = False

    m = d*10 # in the paper 6*(d+5)*maxIt
    mu = 1/(10*maxIt) # in the paper mu = 1/(4*maxIt)
    c_t = 1 # in the paper c_t = 1 / np.sqrt(2 * maxIt * np.power(d+3, 3)), ~ 2e-6
    gamma_t = 1e-3 # in the paper gamma_t = 2*L

    misclassified = []

    print(f'\033[1mParameters: \nxi= {xi},\nmu={mu},\nc_t={c_t},\nm={m},\ngamma_t={gamma_t}\n', flush=True)
    print('\033[0m', flush=True)

    step = 10

    for i in np.arange(0, x_t_chosen.shape[0], step=step):

        print(f'\033[1mWorking on image {i//step+1} of {x_t_chosen.shape[0]//step} with label {i//instances_per_class}', flush=True)
        print('\033[0m', flush=True)
        out_image, class_image = GM_IU(x_t_chosen[i], label_OH_chosen[i], y[i], model, d, L_norm, convex, xi, maxIt, mu, c_t, gamma_t, m)

        if class_image != -1:
            misclassified.append((i, i//instances_per_class, class_image))
            print(f'\nThe misclassified images up to now are {len(misclassified)} out of {i//step+1}: ', misclassified, '\n\n', flush=True)

    return






####################################### Distributed Zeroth-Order Frank Wolfe #######################################


def create_graph(M):

    # create the graph that describes the connections among the M nodes

    # M: number of workers (nodes of the graph)

    nodes = range(M) 

    # create two partitions, S and T. Initially store all nodes in S
    S, T = set(nodes), set() 
    
    # pick a random node, and mark it as visited and as the current node
    current_node = np.random.choice(list(S))
    S.remove(current_node)
    T.add(current_node)

    # create the graph with M nodes
    G = nx.Graph()
    G.add_nodes_from(nodes) 

    
    # create a random connected graph 
    while S:
        # randomly pick the next node from the neighbors of the current node.
        neighbor_node = np.random.choice(nodes)
        # if the new node hasn't been visited, add the edge from current to new
        if neighbor_node not in T:
            edge = (current_node, neighbor_node)
            G.add_edge(*edge)
            S.remove(neighbor_node)
            T.add(neighbor_node)
        # set the new node as the current node. 
        current_node = neighbor_node

    print('Original Graph: \n')

    f = plt.figure(1)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    return G


def add_random_edges(G, M, num):

    # add edges to the graph created previously

    # G: nx.Graph, graph with M nodes
    # M: number of workers (nodes of the graph)
    # num: number of edges we want to add to the graph

    all_edges = [(i,j) for i in range(M) for j in range(i+1,M)]

    counter = 0
    while counter < num and (min(G.degree, key=lambda x: x[1])[1]<9):
        
        current_edges = list(G.edges)
        toBeAdded = all_edges[np.random.randint(len(all_edges))]
        if not((toBeAdded in current_edges) or ((toBeAdded[1], toBeAdded[0]) in current_edges)):
            G.add_edge(*toBeAdded)
            counter += 1
            print(toBeAdded, 'added', flush=True)

    print('\nGraph modified:\n')
    
    f = plt.figure(2)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


def avg_noises(W, G, i, node_images_dict_t, node_images_dict_0):

    # step 3 of the algorithm: Consensus.
    # this function permits to exchange the information on the noise between nodes,
    # obtaining the weighted images from the neighbours 
    
    # W: 2D array, weighting matrix, designed as: W = I-δL (with L=Graph Laplacian)
    # G: nx.Graph, graph with M nodes
    # i: integer, index representing the node
    # node_images_dict_t: dictionary, dictionary with workers as keys and respective perturbed images as values at step t
    # node_images_dict_0: dictionary, dictionary with workers as keys and respective initial images as values
    # noise_t_mean: 3D array, noise computed after the averaging of the information received by the neighbours

    print(f'avg_noises() call, node {i}\n', flush=True)

    noise_t_mean = np.zeros(node_images_dict_t[i].shape[1:]) # 32,32,1
    for j in G.nodes:
        noise_t_mean += W[i,j] * (node_images_dict_t[j][0] - node_images_dict_0[j][0]) # [0] because we only need one noise, and they are all the same in the same worker

    node_images_dict_t[i] = noise_t_mean[np.newaxis, ...] + node_images_dict_0[i]


def perturbate_pixel(node_images_dict_t_i, row, column, c_t):

    # add perturbation along the [row,column] pixel to the images of node i at step t

    # node_images_dict_t_i: 4D array, batch of perturbed images at step t of node i 
    # row: integer, row of the image tensor to be perturbed
    # column: integer, column of the image tensor to be perturbed
    # c_t: float, smoothing parameter
    # images_batch: 4D array, copy of the node_images_dict_t_i
    
    images_batch = node_images_dict_t_i.copy()
    images_batch[:,row,column] += c_t
    
    return images_batch


def gradient_estimation(loss_dict_t, label_OH_dict, node_images_dict_t, grad_t_dict, i, d, c_t, model, loss_plot_dict):

    # step 4 of the algorithm: gradient estimation. It exploit KWSA in order to estimate the gradient at each node i

    # loss_dict_t: dictionary, dictionary with workers as keys and loss of respective perturbed images as values at step t
    # label_OH_dict: dictionary, dictionary with workers as keys and one hot encoded real labels of the instance images
    # node_images_dict_t: dictionary, dictionary with workers as keys and respective perturbed images as values at step t
    # grad_t_dict: dictionary, dictionary with workers as keys and gradient estimated as value at step t
    # i: integer, index representing the node
    # d: integer, dimensionality of the problem
    # c_t: float, smoothing parameter
    
    print(f'gradient_estimation() call, node {i}\n', flush=True)

    loss_dict_t[i] = tf.cast(tf.keras.losses.categorical_crossentropy(label_OH_dict[i], model.predict(node_images_dict_t[i])), dtype=tf.float64)
    #####
    loss_plot_dict[i] = loss_dict_t[i]
    #####
    grad = np.empty((int(np.sqrt(d)), int(np.sqrt(d))))

    for row in range(int(np.sqrt(d))):
        if (row > 0) and (row % 8 == 0):
            print(f'{int(np.sqrt(d)*row)} images perturbed on a single pixel in node {i} up to now\n', flush=True)
        for column in range(int(np.sqrt(d))):

            scalar = (tf.math.reduce_sum((tf.cast(tf.keras.losses.categorical_crossentropy(label_OH_dict[i], 
                model.predict(perturbate_pixel(node_images_dict_t[i],row,column,c_t))), dtype=tf.float64)) - loss_dict_t[i])/c_t)
            grad[row, column] = scalar

    grad_t_dict[i] = grad

def approximate_average_gradient(grad_t_dict, grad_tt_dict, grad_tt_mean_dict, GRAD_t_mean_dict, i): 

    # step 5 of the algorithm (part 1): aggregating. It aggregates the average gradients of step t-1,
    # the gradients at step t and the gradients at step t-1 at each node

    # grad_t_dict: dictionary, dictionary with workers as keys and gradient estimated as value at step t
    # grad_tt_dict: dictionary, dictionary with workers as keys and gradient estimated as value at step t-1
    # grad_tt_mean_dict: dictionary, dictionary with workers as keys and average gradient as value at step t-1
    # GRAD_t_mean_dict: dictionary, dictionary with workers as keys and aggregated gradient as value at step t
    # i: integer, index representing the node

    print(f'approximate_average_gradient() call, node {i}\n', flush=True)

    GRAD_t_mean_dict[i] = grad_tt_mean_dict[i] + grad_t_dict[i] - grad_tt_dict[i]


def avg_gradients(W, G, grad_t_mean_dict, GRAD_t_mean_dict, i):

    # step 5 of the algorithm (part 2): average. This function permits to exchange the information 
    # on the average gradients between nodes, obtaining the weighted gradients from the neighbours 

    # W: 2D array, weighting matrix, designed as: W = I-δL (with L=Graph Laplacian)
    # G: nx.Graph, graph with M nodes
    # grad_t_mean_dict: dictionary, dictionary with workers as keys and average gradient as value at step t
    # GRAD_t_mean_dict: dictionary, dictionary with workers as keys and aggregated gradient as value at step t
    # i: integer, index representing the node

    print(f'avg_gradients() call, node {i}\n', flush=True)

    grad_temp = np.zeros((grad_t_mean_dict[i].shape))

    for j in G.nodes:
        grad_temp += W[i,j]*GRAD_t_mean_dict[j]
    
    grad_t_mean_dict[i] = grad_temp
    

def FW_step(grad_t_mean_dict, node_images_dict_t, node_images_dict_0, v_dict_t, i, xi, L_norm, gamma_t):

    # step 6 of the algorithm: update. It computes the new noise and add it to the images.

    # grad_t_mean_dict: dictionary, dictionary with workers as keys and average gradient as value at step t
    # node_images_dict_t: dictionary, dictionary with workers as keys and respective perturbed images as values at step t
    # node_images_dict_0: dictionary, dictionary with workers as keys and respective initial images as values
    # v_dict_t: dictionary, dictionary with workers as keys and respective images with perturbation at step t as values
    # i: integer, index representing the node
    # xi: float, bound of the perturbation
    # L_norm: string, norm type
    # gamma_t: step-size

    print(f'FW_step() call, node {i}\n', flush=True)

    if L_norm == 'L_inf':
        v_dict_t[i] = xi * np.sign(grad_t_mean_dict[i])[np.newaxis, ..., np.newaxis] + node_images_dict_0[i]
    else:
        raise 'Invalid L_norm string'
    
    node_images_dict_t[i] = (1 - gamma_t) * node_images_dict_t[i] + gamma_t * v_dict_t[i]



def nodeDistributedStep1(tupleContainer1):

    # distribution function: step 3, step 4, step 5 (part 1) of the algorithm

    """
    tupleContainer1 = 
            (keyIdx, # 0
            label_OH_dict,
            loss_dict_t, # 2
            node_images_dict_t,
            node_images_dict_0, # 4
            grad_t_dict,
            grad_tt_dict, # 6
            grad_tt_mean_dict,
            GRAD_t_mean_dict, # 8
            W,
            G, # 10
            d,
            c_t # 12,
            model,
            loss_plot_dict #14
            )
    """

    # step 3 algorithm
    avg_noises(tupleContainer1[9], tupleContainer1[10], tupleContainer1[0], tupleContainer1[3], tupleContainer1[4])
    
    # step 4 algorithm
    gradient_estimation(tupleContainer1[2], tupleContainer1[1], tupleContainer1[3], tupleContainer1[5], tupleContainer1[0], tupleContainer1[11], tupleContainer1[12], tupleContainer1[13], tupleContainer1[14])

    # step 5 algorithm (pt.1)
    approximate_average_gradient(tupleContainer1[5], tupleContainer1[6], tupleContainer1[7], tupleContainer1[8], tupleContainer1[0])

    return

def nodeDistributedStep2(tupleContainer2):

    # distribution function: step 5 (part 2) of the algorithm

    """
    tupleContainer2 = 
            (keyIdx, # 0
            GRAD_t_mean_dict,
            grad_t_mean_dict, # 2
            node_images_dict_t,
            node_images_dict_0, # 4
            v_dict_t,
            xi, # 6
            L_norm,
            W, # 8
            G,
            gamma_t, # 10
            t
            )
    """

    # step 5 algorithm (pt.2)
    avg_gradients(tupleContainer2[8], tupleContainer2[9], tupleContainer2[2], tupleContainer2[1], tupleContainer2[0])


def nodeDistributedStep3(tupleContainer3):

    # distribution function: step 6 of the algorithm

    """
    tupleContainer2 = 
            (keyIdx, # 0
            GRAD_t_mean_dict,
            grad_t_mean_dict, # 2
            node_images_dict_t,
            node_images_dict_0, # 4
            v_dict_t,
            xi, # 6
            L_norm,
            W, # 8
            G,
            gamma_t, # 10
            t
            )
    """

    # step 6 algorithm
    FW_step(tupleContainer3[2], tupleContainer3[3], tupleContainer3[4], tupleContainer3[5], tupleContainer3[0], tupleContainer3[6], tupleContainer3[7], tupleContainer3[10])


def plots_distributed(gradient_dict, misclassified_percentual, loss_plot_list):
    
    #gradient_dict = {1: [lista di 10 gradienti], 2:[lista di 10 gradienti]}
    gradient_dict = {k: np.array(gradient_dict[k]) for k in gradient_dict.keys()}
    #gradient_matrix = [1 riga: 10 gradienti dell'iterazione 1] (maxIt,10,32,32)
    gradient_matrix = np.array([np.array(gradient_dict[k]) for k in gradient_dict.keys()])
    
    diff_norm_gradients = []
    norm_gradients = [np.zeros(10)]
    for grdnts_key in gradient_dict:
        norm_gradients.append(np.linalg.norm(gradient_dict[grdnts_key], axis=(1,2)))
    
    
    diff_norm_gradients = np.linalg.norm(np.diff(gradient_matrix, axis=0), axis=(2,3)) 
      
    norm_gradients = np.array(norm_gradients).T
    
    # Gradient norm vs Iterations
    plt.figure()
    for row in norm_gradients:
        plt.plot(range(len(row)),row)
    plt.ylabel('Gradient norm')
    plt.xlabel('Iterations')
    plt.show()
        
    
    # Difference in norm of the Gradients vs Iterations
    diff_norm_gradients = diff_norm_gradients.T
    
    plt.figure()
    for row in diff_norm_gradients:
        plt.plot(range(len(row)),row)
    plt.ylabel('Difference in norm of the gradients')
    plt.xlabel('Iterations')
    plt.show()

    
    # Number of misclassified images vs Iterations
    misclassified_percentual.insert(0,0)
    plt.plot(np.arange(len(misclassified_percentual)), misclassified_percentual)
    plt.ylabel('Misclassified Images')
    plt.xlabel('Iteration')
    plt.show()
    
    
    # Loss vs iterations
    plt.plot(np.arange(len(loss_plot_list)), loss_plot_list)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()
    

def main_DistrZOFW(model, xT, y_OH):

    # run Distributed Zeroth Order Frank-Wolfe

    # model: keras.engine.sequential.Sequential or keras.engine.training.Model, target model


    # M: integer, number of workers (nodes of the graph)
    # n: integer, number of images in each node
    # d: integer, dimensionality of the problem
    # MaxIt: integer, maximum number of iterations
    # xi: float, bound of the perturbation
    # alpha: float, parameter used in step-size definition
    # L_norm: string, norm type
    # G: nx.Graph, graph with M nodes
    # D: 2D array, diagonal matrix of G
    # A: 2D array, adjacency matrix of G
    # L: 2D array, Laplacian matrix of G
    # W: 2D array, weighting matrix, designed as: W = I-δL (with L=Graph Laplacian)
    # num: integer, number of edges to add to the graph G
    # c_t: float, smoothing parameter
    # gamma_t: float, step-size

    num = 32 
    M = 10
    n = 100
    d = 1024
    MaxIt = 5
    xi = 0.25
    alpha = 0.5 
    L_norm = 'L_inf'
    

    # create graph

    G = create_graph(M)
    # add edges to the graph
    if num > 0:
        add_random_edges(G, M, num)

    max_degree = max(G.degree, key=lambda x: x[1])[1]
    delta = 1/M # max_degree

    D = np.diag([G.degree[i] for i in range(M)])
    A = nx.adjacency_matrix(G).todense()
    L = D - A
    W = np.eye(M) - delta*L

    print(f'Nodes degree: {G.degree} \n')
    print(f'||W-J||: {np.linalg.norm(W - np.ones((M,M))/M)}', flush=True)


    # create image batches

    # choose the indexes
    chosen_indexes = pick_balanced_labels(xT, y_OH, model, 100)

    # pick just 1000 images (they are ordered with respect to the labels)
    x_t_chosen = np.empty((len(chosen_indexes)*len(chosen_indexes[0]), 32, 32, 1))
    label_OH_chosen = np.empty((len(chosen_indexes)*len(chosen_indexes[0]), 10))
    for h in range(len(chosen_indexes)):
        for k in range(len(chosen_indexes[h])):
            x_t_chosen[h*len(chosen_indexes[0]) + k] = xT[chosen_indexes[h][k]] # 1000, 32, 32, 1
            label_OH_chosen[h*len(chosen_indexes[0]) + k] = y_OH[chosen_indexes[h][k]] # 1000, 10

    ######
    gradient_dict = {}
    misclassified_percentual = []
    loss_plot_dict = {}
    loss_plot_list = []
    ######

    # dictionary initialization

    # split dataset among the nodes (fixed)
    node_indexes_dict = {}

    # oscillates between average and actual images at step t
    node_images_dict_t = {}

    label_OH_dict = {}
    loss_dict_t = {}
    y_dict_t = {}

    for i in range(M):

        node_indexes_dict[i] = np.arange(i, n*M, len(label_OH_chosen[0])).astype('int')

        node_images_dict_t[i] = x_t_chosen[node_indexes_dict[i]]
        label_OH_dict[i] = label_OH_chosen[node_indexes_dict[i].astype("int")]
        y_dict_t[i] = model.predict(node_images_dict_t[i])
        loss_dict_t[i] = tf.cast(tf.keras.losses.categorical_crossentropy(label_OH_dict[i], y_dict_t[i]), dtype=tf.float64)

    node_images_dict_0 = node_images_dict_t.copy() # original images dictionary
    y_dict_0 = y_dict_t.copy()

    v_dict_t = {}

    grad_t_dict = {}
    grad_tt_dict = {k:np.zeros((int(np.sqrt(d)),int(np.sqrt(d)))) for k in range(M)}
    grad_t_mean_dict = {k:np.zeros((int(np.sqrt(d)),int(np.sqrt(d)))) for k in range(M)}
    grad_tt_mean_dict = grad_t_mean_dict.copy()
    GRAD_t_mean_dict = {}


    global_tic = time.time()

    
    for t in range(1, MaxIt+1):
        print(f'Iteration {t}\n', flush=True)
        
        iteration_time = time.time()

        gamma_t = 1 / np.power(t, alpha)
        c_t = gamma_t/d
        
        # distribute from here

        startPool1 = time.time()

        # step 3, step 4, step 5 (part 1) of the algorithm
        p = Pool(M)
        arguments1 = tuple(
            (keyIdx,
            label_OH_dict,
            loss_dict_t,
            node_images_dict_t,
            node_images_dict_0,
            grad_t_dict,
            grad_tt_dict,
            grad_tt_mean_dict,
            GRAD_t_mean_dict,
            W,
            G,
            d,
            c_t,
            model,
            loss_plot_dict
            ) for keyIdx in range(M))
        _ = p.map(nodeDistributedStep1, arguments1) 
        

        print(f'\nPool1 CPU time (avg_noise(), gradient_estimation(), approximate_average_gradient() calls): {time.time() - startPool1}\n', flush=True)

        ####
        loss_plot_list.append(np.mean(list(loss_plot_dict.values())))
        ####
        
        startPool2 = time.time()

        # step 5 (part 2) of the algorithm
        p = Pool(M) 
        arguments2 = tuple(
            (keyIdx,
            GRAD_t_mean_dict,
            grad_t_mean_dict,
            node_images_dict_t,
            node_images_dict_0,
            v_dict_t,
            xi,
            L_norm,
            W,
            G,
            gamma_t,
            t
            ) for keyIdx in range(M))
        _ = p.map(nodeDistributedStep2, arguments2)
        
        
        print(f'\nPool2 CPU time (avg_gradients() call): {time.time() - startPool2}\n', flush=True)
        
        
        startPool3 = time.time()
        
        # step 6 of the algorithm
        p = Pool(M) 
        arguments3 = tuple(
            (keyIdx,
            GRAD_t_mean_dict,
            grad_t_mean_dict,
            node_images_dict_t,
            node_images_dict_0,
            v_dict_t,
            xi,
            L_norm,
            W,
            G,
            gamma_t,
            t
            ) for keyIdx in range(M))
        _ = p.map(nodeDistributedStep3, arguments3)
        

        print(f'\nPool3 CPU time (FW_step() call): {time.time() - startPool3}\n', flush=True)


        # print the new noise, the total noise and the instance images perturbed at step t

        for i in range(M):

            print(f'\nNew noise at step {t} in node {i}', flush=True)
            f = plt.figure(100+i) # translation of the plot index, not meaningful
            plt.imshow((v_dict_t[i][0] - node_images_dict_0[i][0]).squeeze(), vmin = 0, vmax = 1)
            plt.show()

            print(f'Total noise at step {t} in node {i}', flush=True)
            f = plt.figure(1000+i) # translation of the plot index, not meaningful
            plt.imshow((node_images_dict_t[i][0] - node_images_dict_0[i][0]).squeeze(), vmin = 0, vmax = 1) # - x_0^i_mean (which we don't save)
            plt.show()
            
            print(f'Example of image perturbed at step {t} in node {i}', flush=True)
            f = plt.figure(10000+i) # translation of the plot index, not meaningful
            plt.imshow((node_images_dict_t[i][0]).squeeze(), vmin = 0, vmax = 1) # - x_0^i_mean (which we don't save)
            plt.show()


        # update the dictionaries

        grad_tt_dict = grad_t_dict.copy()
        grad_tt_mean_dict = grad_t_mean_dict.copy()

        #####
        gradient_dict[t] = []
        for key in grad_tt_mean_dict.keys():
            gradient_dict[t].append(grad_tt_mean_dict[key])
        #####
        
        print(f'Iteration CPU elapsed time: {time.time()-iteration_time}\n')
        # count the misclassified images at step t
        
        counter_miscl = 0
        for i in range(M):
            
            totalMiscl = np.arange(node_images_dict_0[i].shape[0])[np.apply_along_axis(np.argmax, 1, label_OH_dict[i]) != np.apply_along_axis(np.argmax, 1, model.predict(node_images_dict_t[i]))]
            
            print(f'Miscl. by perturbation by worker {i}: {len(set(totalMiscl))}', flush=True)
            counter_miscl += len(set(totalMiscl))
        print('\n')
        
        #####
        non_misclassified_images = {}
        for i in range(M):
            pred_labels = model.predict(node_images_dict_t[i])
            true_labels = label_OH_dict[i]
            for idx,elem in enumerate(label_OH_dict[i]):
                true_label = np.argmax(true_labels[idx])
                pred_label = np.argmax(pred_labels[idx])
                if true_label == pred_label:
                    if true_label in non_misclassified_images.keys():
                        non_misclassified_images[true_label] += 1
                    else:
                        non_misclassified_images[true_label] = 1
                        
        print(f'Number of images not misclassified per class:\n {non_misclassified_images}')
                    
        #####
        
        
        #####
        misclassified_percentual.append(counter_miscl)
        #####
        
        # exit condition (misclassified more than 80% of the images)
        if (counter_miscl/x_t_chosen.shape[0])>0.8:
            print("Misclassified more than 80% of the images. Stop!")
            # last step: average the perturbed images
            for i in range(M):
                avg_noises(W, G, i, node_images_dict_t, node_images_dict_0)
                for rapr in range(M):
                    f = plt.figure(10000+100*i+rapr) # translation of the plot index, not meaningful
                    plt.imshow(node_images_dict_t[i][rapr*10].squeeze(), vmin = 0, vmax = 1)
                    plt.show()
            print('Total elapsed CPU time:', time.time() - global_tic, flush=True)
            #plots
            plots_distributed(gradient_dict, misclassified_percentual, loss_plot_list)
            return 

    # last step: average the perturbed images
    for i in range(M):
        avg_noises(W, G, i, node_images_dict_t, node_images_dict_0)
        for rapr in range(M):
            f = plt.figure(10000+100*i+rapr) # translation of the plot index, not meaningful
            plt.imshow(node_images_dict_t[i][rapr*10].squeeze(), vmin = 0, vmax = 1)
            plt.show()
        totalMiscl = np.arange(node_images_dict_0[i].shape[0])[np.apply_along_axis(np.argmax, 1, label_OH_dict[i]) != np.apply_along_axis(np.argmax, 1, model.predict(node_images_dict_t[i]))]
        print(f'Miscl. by perturbation worker {i}: {len(set(totalMiscl))}', flush=True)
        

    print('Total elapsed CPU time:', time.time() - global_tic, flush=True)
    #plots
    plots_distributed(gradient_dict, misclassified_percentual, loss_plot_list)
    return 






####################################### Decentralized Variance Reduced Zeroth-Order Frank Wolfe #######################################

def deterministic_gradient(S1_node_images, S1_node_labels_OH, S1_node_loss_t, canonical_basis, c_t_KWSA, dim, model):

    # compute the gradient estimation with KWSA technique

    # S1_node_images: 4D array, perturbed images batch at step t at node i
    # S1_node_labels_OH: 2D array, one hot encoded real labels of the images batch at node i
    # S1_node_loss_t: 1D array, loss of the perturbed images batch at step t at node i
    # canonical_basis: 1D array, zeros vector with dimensions of an image
    # c_t_KWSA: float, smoothing parameter (KWSA case)
    # dim: integer, dimensionality of the problem

    if dim%256 == 0:
        print(f"Computing deterministic gradient estimation for dim: {dim}")

    # modify the canonical basis in position j
    canonical_basis[dim] = 1
    # reshape the canonical basis as the shape of an image
    canonical_basis = canonical_basis.reshape((1,S1_node_images.shape[1],S1_node_images.shape[2],1)) 

    # add perturbation along the canonical bases to the images
    S1_node_images += (canonical_basis*c_t_KWSA)
    # compute the loss on the perturber images (100,)
    S1_node_loss_perturbed_t = tf.cast(CCE(S1_node_labels_OH, model.predict(S1_node_images)),dtype=tf.float64)
    
    # remove perturbation along the canonical bases to the images
    S1_node_images -= (canonical_basis*c_t_KWSA)

    # compute the j-th component of the gradient (100,)
    grad_i_dim_t = (S1_node_loss_perturbed_t - S1_node_loss_t)/(c_t_KWSA)

    grad_i_dim_t = np.sum(grad_i_dim_t)/(S1_node_images.shape[0]) 

    # restore the canonical basis
    canonical_basis = canonical_basis.reshape((-1))
    canonical_basis[dim] = 0

    return grad_i_dim_t



def stochastic_gradient(S2_node_images_t, S2_node_labels_OH_t, S2_node_loss_t, 
                        S2_node_images_tt, S2_node_loss_tt,
                        S2_node_directions, c_t_RDSA, i, model):

    # compute the gradient estimation with RDSA technique

    # S2_node_images_t: 4D array, perturbed images batch at step t at node i
    # S2_node_labels_OH_t: 2D array, one hot encoded real labels of the images batch at node i
    # S2_node_loss_t: 1D array, loss of the perturbed images batch at step t at node i
    # S2_node_images_tt: 4D array, perturbed images batch at step t-1 at node i
    # S2_node_loss_tt: 1D array, loss of the perturbed images batch at step t-1 at node i
    # S2_node_directions: 4D array, gaussian directions 
    # c_t_RDSA: float, smoothing parameter (RDSA case)
    # i: integer, node index
    
    grad_i_dir_t = np.zeros(S2_node_images_t.shape)

    print(f"Computing stochastic gradient estimation in node: {i}")
    # add perturbation along the gaussian direction to the images at step t
    S2_node_images_t += (S2_node_directions*c_t_RDSA)
    # compute the loss at step t on the perturbed images
    S2_node_loss_perturbed_t = tf.cast(CCE(S2_node_labels_OH_t, model.predict(S2_node_images_t)),dtype=tf.float64)
    # add perturbation along the gaussian direction to the images at step t-1
    S2_node_images_tt += (S2_node_directions*c_t_RDSA)
    # compute the loss at step t on the perturber images
    S2_node_loss_perturbed_tt = tf.cast(CCE(S2_node_labels_OH_t, model.predict(S2_node_images_tt)),dtype=tf.float64)


    # compute the gradient
    first_term = (S2_node_loss_perturbed_t - S2_node_loss_t)/(c_t_RDSA) 
    second_term = (S2_node_loss_perturbed_tt - S2_node_loss_tt)/(c_t_RDSA) 
    
    for term in range(S2_node_images_t.shape[0]):
        grad_i_dir_t[term] = (first_term[term]-second_term[term])*S2_node_directions[term]
    

    grad_i_t = np.sum(grad_i_dir_t,axis=0)/S2_node_images_t.shape[0]
    grad_i_t = grad_i_t.squeeze() 

    return grad_i_t


def plots_decentralized(gradient_list, misclassified_percentual, loss_plot_list):
    
    norm_gradients = [0]
    for grdnt in gradient_list:
        norm_gradients.append(np.linalg.norm(grdnt))
        
    # Gradient norm vs Iterations
    plt.plot(np.arange(len(norm_gradients)),norm_gradients)
    plt.ylabel('Gradient norm')
    plt.xlabel('Iteration')
    plt.show()
    
    
    # Difference in norm of the Gradients vs Iterations
    diff_norm_gradients = [0]
    for idx in range(len(gradient_list)-1):
        diff_norm_gradients.append(np.linalg.norm(gradient_list[idx+1]-gradient_list[idx]))
    
    plt.plot(np.arange(len(diff_norm_gradients)),diff_norm_gradients)
    plt.ylabel('Norm Gradient Difference')
    plt.xlabel('Iteration')
    plt.show()
    
    
    # Number of misclassified images vs Iterations
    misclassified_percentual.insert(0,0)
    plt.plot(np.arange(len(misclassified_percentual)),misclassified_percentual)
    plt.ylabel('Image Misclassified')
    plt.xlabel('Iteration')
    plt.show()
    
    
    # Loss vs Iterations
    plt.plot(np.arange(len(loss_plot_list)), loss_plot_list)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()


def main_DecentrVRZOFW(xT, y_OH, loss, model):
    
    # run Decentralized Variance Reduced Zeroth Order Frank-Wolfe

    # model: keras.engine.sequential.Sequential or keras.engine.training.Model, target model


    # M: integer, number of workers 
    # n0: float, parameter ([1, sqrt(M*n)/6])
    # n: integer, number of images in each node
    # xi: float, bound of the perturbation
    # maxIt: integer, maximum number of iterations
    # stepsize: float
    # q: integer, module parameter for if condition (n0*sqrt(Mn)/6)
    # d: integer, dimensionality of the problem
    # L_norm: string, norm type
    # S1: integer, Mnd
    # S2: integer, (2d+9)*sqrt(Mn)/n0
    # canonical_basis: 1D array, zeros vector with dimensions of an image
    # c_t_KWSA: float, smoothing parameter (KWSA case)
    # c_t_RDSA: float, smoothing parameter (RDSA case)

    
    # choose the indexes
    chosen_indexes = pick_balanced_labels(xT, y_OH, model, 100)

    # pick just 1000 images (they are ordered with respect to the labels)
    x_t = np.empty((len(chosen_indexes)*len(chosen_indexes[0]), 32, 32, 1))
    label_OH = np.empty((len(chosen_indexes)*len(chosen_indexes[0]), 10))
    loss_0 = np.empty((len(chosen_indexes)*len(chosen_indexes[0]), 1))
    for i in range(len(chosen_indexes)):
        for j in range(len(chosen_indexes[i])):
            x_t[i*len(chosen_indexes[0]) + j] = xT[chosen_indexes[i][j]]
            label_OH[i*len(chosen_indexes[0]) + j] = y_OH[chosen_indexes[i][j]]
            loss_0[i*len(chosen_indexes[0]) + j] = loss[chosen_indexes[i][j]]
    loss_0 = tf.cast(tf.convert_to_tensor(loss_0.reshape(-1)),dtype=tf.float64)
    

    y = model.predict(x_t[..., np.newaxis])

    # constant definition
    M = 10
    n0 = 5 # in [1, 5.5]
    n = x_t.shape[0]/M
    xi = 0.25
    maxIt = 5
    stepsize = 1 / (maxIt**(3/4))
    q = int(n0 * np.sqrt(M*n) / 6) # 26
    d = 1024
    L_norm = 'L_inf'
    S1 = M*n*d
    S2 = int((2*d+9)*np.sqrt(M*n)/n0)


    # arrays initialization
    x0 = x_t.copy()
    x_tt = x_t.copy()
    v_t = x_t.copy() 
    ####
    loss_plot_list = []
    ####
    loss_t = tf.identity(loss_0) #(1000,)
    loss_tt = tf.identity(loss_t) #(1000,)
    
    grad_i_fixed_t = np.empty((M, x_t.shape[1], x_t.shape[2]))
    grad_i_fixed_tt = np.empty((M, x_t.shape[1], x_t.shape[2]))
    grad_t = np.empty((x_t.shape[1],x_t.shape[2]))

    canonical_basis = np.zeros(d)
    ######
    gradient_list = []
    misclassified_percentual = []
    misclassified_images = []
    ######
    
    # split dataset among the nodes (fixed)
    for i in range(M):
        exec(f'node_indexes_{i}=np.arange(i, n*M, len(label_OH[0]))')
        exec(f'label_OH_{i} = label_OH[node_indexes_{i}.astype("int")]')
    
    global_tic = time.time()
    
    
    for t in range(maxIt):
        
        print('Iterazione n.', t+1)

        c_t_KWSA = 2/(np.sqrt(d)*np.power(t+8,1/3)).astype('longdouble')
        c_t_RDSA = (2*np.sqrt(S2) / (np.power(d, 3/2) * np.power(t+8, 1/3))).astype('longdouble')

        # split images and losses among workers
        for i in range(M):
            node_indexes_i = eval(f'node_indexes_{i}').astype("int")
            x_t_i = x_t[node_indexes_i]
            exec(f'x_t_{i} = x_t_i')
            loss_t_i = tf.convert_to_tensor([loss_t[idx] for idx in node_indexes_i]) 
            exec(f'loss_t_{i} = loss_t_i')

        # if step
        if t%q == 0:

            print("I'm in if")

            tic = time.time()

            # initialize gradient at step t at worker i
            grad_i_fixed_t = np.empty((M, x_t.shape[1], x_t.shape[2]))

            for i in range(M): 
                
                node_time = time.time()
                
                S1_node_images = eval(f'x_t_{i}') #images with index S1_node_indexes
                S1_node_labels_OH = eval(f'label_OH_{i}') #Labels One-Hot-Encoded of the images in S1_node
                S1_node_loss_t = eval(f'loss_t_{i}') #Loss of the images with the same indexes extracted for S1_node 
            
                grad_i_DIM_t = np.empty(d)
                for dim in range(d): #for each dimension
                    grad_i_DIM_t[dim] = deterministic_gradient(S1_node_images, S1_node_labels_OH, S1_node_loss_t, canonical_basis, c_t_KWSA, dim, model)
                grad_i_DIM_t = grad_i_DIM_t.reshape(32,32)

                print(f'Ended node: {i}')
                print(f'Node {i} elapsed CPU time: {time.time()-node_time}')

                grad_i_fixed_t[i] = grad_i_DIM_t 
            grad_i_fixed_tt = grad_i_fixed_t.copy()
            
        # else step
        else:

            print("I'm in else")

            tic = time.time()
            
            grad_I_t = np.empty((M, x_t.shape[1], x_t.shape[2]))
            grad_i_fixed_t = np.empty((M, x_t.shape[1], x_t.shape[2]))

            for i in range(M):
                
                node_time = time.time()
                
                S2_node_indexes = np.random.choice(eval(f'node_indexes_{i}'), S2, replace=True).astype('int')

                S2_node_directions = np.random.standard_normal((int(S2), *x_t.shape[1:]))
                

                S2_node_images_t = x_t[S2_node_indexes] 
                S2_node_labels_OH_t = label_OH[S2_node_indexes] 
                S2_node_loss_t = tf.convert_to_tensor([loss_t[idx] for idx in S2_node_indexes])
                
                
                S2_node_images_tt = x_tt[S2_node_indexes] 
                S2_node_loss_tt = tf.convert_to_tensor([loss_tt[idx] for idx in S2_node_indexes]) 
                
                grad = stochastic_gradient(S2_node_images_t, S2_node_labels_OH_t, S2_node_loss_t, S2_node_images_tt, S2_node_loss_tt, S2_node_directions, c_t_RDSA, i, model)
        
                grad_I_t[i] = grad 
                
                print(f'Node {i} elapsed CPU time: {time.time()-node_time}')


            grad_i_fixed_t = grad_I_t + grad_i_fixed_tt 

            # save previous gradient
            grad_i_fixed_tt = grad_i_fixed_t.copy()


        grad_t = (np.sum(grad_i_fixed_t, axis=0))/M 
        
        ######
        gradient_list.append(grad_t)
        ######

        if L_norm == 'L_inf':
            # for norm l-infinity
            v_t = xi * np.sign(grad_t[np.newaxis,...,np.newaxis]) + x0 
            print('Perturbation:\n')
            plt.imshow((xi * np.sign(grad_t[np.newaxis,...,np.newaxis])).squeeze(),vmin = 0, vmax = 1)
            plt.show()
        else:
            print('Invalid L-norm string')
            return

        x_tt = x_t.copy()

        # update the image
        x_t = x_t - stepsize * (x_t - v_t)
        print('Avg. perturbation per pixel after convex combination', np.mean(x_t-x0))

        print(f'Iteration {t+1} elapsed CPU time: ', time.time()-tic)

        # new prediction
        y = model.predict(x_t)
        
        # compute the loss of x_t at step t
        loss_tt = tf.identity(loss_t)
        loss_t = tf.cast(CCE(label_OH, y),dtype=tf.float64)
        totalMiscl = np.arange(x0.shape[0])[np.apply_along_axis(np.argmax, 1, label_OH) != np.apply_along_axis(np.argmax, 1, y)]
        print(f'Misclassified at the end: {len(totalMiscl)}')
        ######
        loss_plot_list.append(np.sum(loss_t))
        ######
        ######
        misclassified_percentual.append(len(totalMiscl))
        ######
        
        print('Perturbed images:\n')
        plt.figure(num=10)
        for img in np.arange(0,1000,111):   
            plt.imshow(x_t[img].squeeze(), vmin = 0, vmax = 1)
            plt.show()
            
        #####    
        non_misclassified_images = {}
        for idx,elem in enumerate(label_OH):
            true_label = np.argmax(label_OH[idx])
            pred_label = np.argmax(y[idx])
            if true_label == pred_label:
                if true_label in non_misclassified_images.keys():
                    non_misclassified_images[true_label] += 1
                else:
                    non_misclassified_images[true_label] = 1
        
        
        print(f'Number of images not misclassified per class:\n {non_misclassified_images}')
        #####
        
        if len(totalMiscl)/x_t.shape[0] > 0.7:
            print("Misclassified more than 70% of the images. Stop!")
            totalMiscl = np.arange(x0.shape[0])[np.apply_along_axis(np.argmax, 1, label_OH) != np.apply_along_axis(np.argmax, 1, y)]
            print(f'Misclassified at the end: {len(totalMiscl)}')
            print('Total elapsed CPU time:', time.time() - global_tic)
            #Plots
            plots_decentralized(gradient_list, misclassified_percentual, loss_plot_list)
            return 


    totalMiscl = np.arange(x0.shape[0])[np.apply_along_axis(np.argmax, 1, label_OH) != np.apply_along_axis(np.argmax, 1, y)]
    print(f'Misclassified at the end: {len(totalMiscl)}')
    print('Total elapsed CPU time:', time.time() - global_tic)

    #plots
    plots_decentralized(gradient_list, misclassified_percentual, loss_plot_list)
    
    return 