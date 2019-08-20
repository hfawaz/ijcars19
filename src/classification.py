import numpy as np
import random
import time
from itertools import chain
from keras.models import Model
from keras.utils import np_utils
import keras
from keras import regularizers
import os
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import pandas as pd
import collections
import re
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import sys

matplotlib.use('pdf')
import matplotlib.pyplot as plt


def create_dir(out_dir):
    if os.path.exists(out_dir):
        return None
    else:
        try:
            os.makedirs(out_dir)
        except:
            # in case another machine created the path meanwhile
            return None
        return out_dir


def getExpertiseLevelOfSurgery(surgery_name):
    ## function getMetaDataForSurgeries should be already called
    if surgeries_metadata.__contains__(surgery_name):
        return surgeries_metadata[surgery_name][0]
    return None


def getMetaDataForSurgeries(surgery_type):
    surgeries_metadata = {}
    file = open(root_dir + surgery_type + '_kinematic/' + 'meta_file_' + surgery_type + '.txt', 'r')
    for line in file:
        line = line.strip()  ## remove spaces

        if len(line) == 0:  ## if end of file
            break

        b = line.split()
        surgery_name = b[0]
        expertise_level = b[1]
        b = b[2:]
        scores = [int(e) for e in b]
        surgeries_metadata[surgery_name] = (expertise_level, scores)
    return surgeries_metadata


def fit_encoder(y_train, y_test, y_val):
    y_train_test_val = y_train + y_test + y_val
    encoder.fit(y_train_test_val)


def convertStringClassesToBinaryClasses(y_train, y_test, y_val):
    idx_y_test = len(y_train)
    idx_y_val = len(y_train) + len(y_test)
    y_train_test_val = y_train + y_test + y_val
    y_train_test_val = encoder.transform(y_train_test_val)
    y_train_test_val = np_utils.to_categorical(y_train_test_val)
    y_train = y_train_test_val[0:idx_y_test]
    y_test = y_train_test_val[idx_y_test:idx_y_val]
    y_val = y_train_test_val[idx_y_val:]
    return y_train, y_test, y_val


def get_user_name_and_trial_num(surgery_name, surgery_type):
    user_name = surgery_name.replace(surgery_type + '_', "")[0]
    trial_num = surgery_name.replace(surgery_type + '_', "")[-1]
    return user_name, trial_num


def readFile(file_name, dtype, columns_to_use=None):
    X = np.loadtxt(file_name, dtype, usecols=columns_to_use)
    return X


def generateMaps(surgery_type):
    path = root_dir + surgery_type + '_kinematic' + '/kinematics/AllGestures/'
    for subdir, dirs, files in os.walk(path):
        for file_name in files:
            surgery = readFile(path + file_name, float, columns_to_use=dimensions_to_use)
            surgery_name = file_name[:-4]
            expertise_level = getExpertiseLevelOfSurgery(surgery_name)
            if expertise_level is None:
                continue
            mapSurgeryDataBySurgeryName[surgery_name] = surgery
            mapExpertiseLevelBySurgeryName[surgery_name] = expertise_level
            generateGesturesForSurgery(surgery_name, surgery_type)
    return None


def save_evaluation(out_dir, macro, micro, precision, macro_std, precision_std, val_loss):
    df = pd.DataFrame(data=np.zeros(shape=(1, 6), dtype=np.float32), index=[0],
                      columns=['macro', 'micro', 'precision', 'macro_std', 'precision_std', 'val_loss'])
    df['macro'] = macro
    df['micro'] = micro
    df['precision'] = precision
    df['macro_std'] = macro_std
    df['precision_std'] = precision_std
    df['val_loss'] = val_loss
    df.to_csv(out_dir + 'df_metrics.csv', index=False)


def generateGesturesForSurgery(surgery_name, surgery_type):
    surgery = mapSurgeryDataBySurgeryName[surgery_name]
    path = root_dir + surgery_type + '_kinematic/transcriptions/'
    file_name = surgery_name + '.txt'
    data = readFile(path + file_name, str)
    gestures = []

    for row in data:
        start_index = int(row[0]) - 1
        end_index = int(row[1])
        gesture_name = row[2]
        gestures.append((gesture_name, surgery[start_index:end_index], start_index, end_index))

    mapGesturesBySurgeryName[surgery_name] = gestures
    return True


# shuffles train and labels
def shuffle(x_train, y_train):
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(y_train), 1)
    x_train = x_train.reshape(len(x_train), 1)
    x_y_train = np.concatenate((x_train, y_train), axis=1)
    np.random.shuffle(x_y_train)
    return x_y_train[:, 0], x_y_train[:, 1].tolist()


def validation(surgery_type='Suturing', balanced='Balanced', shuff=True,
               classification='GestureClassification', validation='SuperTrialOut',
               levelClassify=False, val_split=False):
    # levelClassify is used to do a validation on Novic - Intermediate - Expert
    path = path_to_configurations + surgery_type + '/' + balanced + '/' + classification + '/' + validation
    for it in range(1):
        for subdir, dirs, files in os.walk(path):
            # One configuration with two files Train.txt and Test.txt
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            x_val = []
            y_val = []
            user_added_to_val = None
            trial_added_to_val = None
            min_length_train = np.iinfo(np.int32).max  # this is the minimum length of a training instance
            min_length_test = np.iinfo(np.int32).max  # this is the minimum length of a test instance
            min_length_val = np.iinfo(np.int32).max  # this is the minimum length of a val instance
            for file_name in files:
                data = readFile(subdir + '/' + file_name, str)
                surgeries_set = set()
                for gesture in data:
                    surgery_name = find_pattern(gesture[0], surgery_type + '_.00.')
                    surgeries_set.add(surgery_name)

                for surgery_name in surgeries_set:
                    user_name, trial_num = get_user_name_and_trial_num(surgery_name, surgery_type)
                    if file_name == 'Train.txt':
                        if (user_added_to_val is None):
                            user_added_to_val = user_name
                        if (trial_added_to_val is None):
                            trial_added_to_val = trial_num
                        if levelClassify == True:
                            if ((validation == 'SuperTrialOut' and trial_num == trial_added_to_val) and val_split):
                                # we should add to validation set
                                min_length_val = min(len(mapSurgeryDataBySurgeryName[surgery_name]), min_length_val)
                                x_val.append(mapSurgeryDataBySurgeryName[surgery_name])
                                y_val.append(mapExpertiseLevelBySurgeryName[surgery_name])
                            else:  # we add to the train set
                                min_length_train = min(len(mapSurgeryDataBySurgeryName[surgery_name]), min_length_train)
                                x_train.append(mapSurgeryDataBySurgeryName[surgery_name])
                                y_train.append(mapExpertiseLevelBySurgeryName[surgery_name])
                    else:
                        if levelClassify == True:
                            min_length_test = min(len(mapSurgeryDataBySurgeryName[surgery_name]), min_length_test)
                            x_test.append(mapSurgeryDataBySurgeryName[surgery_name])
                            y_test.append(mapExpertiseLevelBySurgeryName[surgery_name])
            # end of one file Train or Test
            if (len(files) > 0):

                x_train = np.array(x_train)
                x_test = np.array(x_test)
                if val_split:
                    x_val = np.array(x_val)

                fit_encoder(y_train, y_test, y_val)

                for itrr in range(max_iterations):

                    # get the hyperparameters
                    architecture = random.choice(architectures)
                    reg = random.choice(regs)
                    lr = random.choice(lrs)
                    filters = int(random.choice(filterss))
                    kernel_size = int(random.choice(kernel_sizes))
                    amsgrad = random.choice(amsgrads)

                    out_dir = out_root_dir + subdir[3:] + '/' \
                              + 'architecture__' + architecture + '/' \
                              + 'reg__' + str(reg) + '/' \
                              + 'lr__' + str(lr) + '/' \
                              + 'filters__' + str(filters) + '/' \
                              + 'kernel_size__' + str(kernel_size) + '/' \
                              + 'amsgrad__' + str(amsgrad) + '/'

                    test_dir = create_dir(out_dir)
                    if (test_dir is None):
                        itrr = itrr - 1
                        continue

                    keras.backend.clear_session()

                    build_model = fcn_each_dim_build_model

                    model = build_model(input_shapes, filters, kernel_size, lr, amsgrad, summary=False, reg=reg)

                    # save init parameters
                    model.save(out_dir + 'model_init.hdf5')

                    epochs_loss, y_test_binary, val_loss = fitModel(model, x_train, y_train,
                                                                    x_test, y_test, x_val,
                                                                    y_val, out_dir, shuff=shuff, val_split=val_split)

                    model = load_model(out_dir + 'model_best.hdf5')

                    # evaluate model and get results for confusion matrix
                    (macro, micro, precision, macro_std, precision_std) = evaluateModel(model, x_test, y_test_binary)
                    save_evaluation(out_dir, macro, micro, precision, macro_std, precision_std, val_loss)


def find_pattern(word, pattern):
    return re.search(r'' + pattern, word).group(0)


def compute_micro(matrix):
    return sum(matrix.diagonal()) / np.sum(matrix)


def compute_macro(matrix):
    res = matrix.diagonal() / np.sum(matrix, axis=1)
    return np.nansum(res) / float(nb_classes)


def compute_macro_std(macro, matrix):
    variance = np.nansum(np.square(matrix.diagonal() / np.sum(matrix, axis=1) - macro)) / float(nb_classes - 1)
    return math.sqrt(variance)


def compute_precision_std(macro, matrix):
    variance = np.nansum(np.square(matrix.diagonal() / np.sum(matrix, axis=0) - macro)) / float(nb_classes - 1)
    return math.sqrt(variance)


def compute_precision(matrix):
    res = matrix.diagonal() / np.sum(matrix, axis=0)
    return np.nansum(res) / float(nb_classes)


def fitModel(model, x_train, y_train, x_test, y_test, x_val, y_val, out_dir, shuff=True, val_split=False):
    # x_test and y_test are used to monitor the overfitting / underfitting not for training
    epochs_loss = "train_loss,test_loss\n"
    # minimum epoch loss on val set
    min_val_loss = -1
    # train for many epochs as specified by nb_epochs
    for epoch in range(0, nb_epochs):
        # shuffle before every epoch training
        if (shuff == True):
            x_train, y_train = shuffle(x_train, y_train)
        # convert string labels to binary forms
        y_train_binary, y_test_binary, y_val_binary = convertStringClassesToBinaryClasses(y_train, y_test, y_val)
        # train each sequence alone
        epoch_train_loss = 0
        epoch_val_loss = 0
        for sequence, label in zip(x_train, y_train_binary):
            loss, acc = model.train_on_batch(split_input_for_training(sequence), label.reshape(1, nb_classes))
            epoch_train_loss += loss  ################# change if monitor acc instead of loss

        epoch_train_loss = epoch_train_loss / len(x_train)
        if val_split:
            epoch_val_loss = evaluate_for_epoch(model, x_val, y_val_binary)
            if (epoch_val_loss < min_val_loss or min_val_loss == -1):
                # this is to choose finally the model that yields the best results on the validation set
                model.save(out_dir + 'model_best.hdf5')
                min_val_loss = epoch_val_loss
        else:  # we evaluate on the train
            if (epoch_train_loss < min_val_loss or min_val_loss == -1):
                # this is to choose finally the model that yields the best results on the validation set
                model.save(out_dir + 'model_best.hdf5')
                min_val_loss = epoch_train_loss

        model.save(out_dir + 'model_curr.hdf5')

        epochs_loss += str(epoch_train_loss) + ',' + str(epoch_val_loss) + '\n'

    return epochs_loss, y_test_binary, min_val_loss


def evaluate_for_epoch(model, x_test, y_test):
    epoch_test_loss = 0
    for test, label in zip(x_test, y_test):
        loss, acc = model.evaluate(split_input_for_training(test), label.reshape(1, nb_classes), verbose=0)
        epoch_test_loss += loss  ############### change if monitor loss instead of accuracy
    return epoch_test_loss / len(x_test)


def evaluateModel(model, x_test, y_test_binary):
    confusion_matrix_f = pd.DataFrame(np.zeros(shape=(nb_classes, nb_classes)), index=classes, columns=classes)

    for test, label in zip(x_test, y_test_binary):
        model.evaluate(split_input_for_training(test), label.reshape(1, nb_classes), verbose=0)
        p = model.predict(split_input_for_training(test), batch_size=1)
        predicted_integer_label = np.argmax(p).astype(int)
        predicted_label = encoder.inverse_transform([predicted_integer_label])[0]
        correct_label = encoder.inverse_transform([np.argmax(label)])[0]
        confusion_matrix[correct_label][predicted_label] += 1.0
        confusion_matrix_f[correct_label][predicted_label] += 1.0

    matrix_f = confusion_matrix_f.values
    macro = compute_macro(matrix_f)
    return (macro, compute_micro(matrix_f), compute_precision(matrix_f)
            , compute_macro_std(macro, matrix_f), compute_precision_std(macro, matrix_f))


def cas(idx_to_explain=0):
    # idx_to_explain corresponds to the id of the class to explain
    generateMaps(surgery_type)

    model = keras.models.load_model('model-calssification-example.h5')

    surgery_name = surgery_type + '_E002'

    time_series_original = mapSurgeryDataBySurgeryName[surgery_name]

    w_k_c = model.layers[-1].get_weights()[0]  # weights for each filter k for each class c

    new_input_layer = model.inputs  # same input of the original model

    new_outpu_layer = [model.get_layer("conv_final").output,
                       model.layers[-1].output]  # output is both the original as well as the before last layer

    new_function = keras.backend.function(new_input_layer, new_outpu_layer)

    new_feed_forward = new_function

    [conv_out, predicted] = new_feed_forward(split_input_for_training(time_series_original))

    print("predicted_label:" + str(np.argmax(predicted)))

    cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))

    conv_out = conv_out[0, :, :]

    for k, w in enumerate(w_k_c[:, idx_to_explain]):
        cas += w * conv_out[:, k]

    minimum = np.min(cas)
    if minimum < 0:
        cas = cas + abs(minimum)
    else:
        cas = cas - minimum

    cas = cas / max(cas)
    cas = cas * 100
    cas = cas.astype(int)

    x_master_left = time_series_original[:, 0]
    y_master_left = time_series_original[:, 1]
    z_master_left = time_series_original[:, 2]

    fig = plt.figure()
    plot3d = fig.add_subplot(111, projection='3d')
    pltmap = plot3d.scatter(x_master_left, y_master_left, z_master_left,
                            c=cas, cmap='jet', s=5, linewidths=0)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    clbr = plt.colorbar(pltmap)
    clbr.set_ticks([])
    plt.savefig(out_root_dir + surgery_name + '.pdf')


# the sequence variable is the multivariate time series or in this case the surgical task
# we want to split the inputs in order to train
def split_input_for_training(sequence):
    # get number of hands
    num_hands = len(input_shapes)
    # get number of dimensions cluster for each hand
    num_dim_clusters = len(input_shapes[0])
    # define the new input sequence
    x = []
    # this is used to keep track of the assigned dimensions
    last = 0
    # loop over each hand
    for i in range(num_hands):
        # loop for each hand over the cluster of dimensions
        for j in range(num_dim_clusters):
            # assign new input same length but different dimensions each time
            x.append(np.array([sequence[:, last:(last + input_shapes[i][j][1])]]))
            # remember last assigned
            last = input_shapes[i][j][1]
    # return the new input
    return x


def fcn_each_dim_build_model(input_shapes, filters, kernel_size, lr, amsgrad, summary=False, reg=0.01):
    # get number of hands
    num_hands = len(input_shapes)
    # get number of dimensions cluster for each hand
    num_dim_clusters = len(input_shapes[0])
    # first index for hand second for  dims
    x = [[None for a in range(0, num_dim_clusters)] for b in range(num_hands)]
    # first conv layer on each dim cluster for each hand
    conv1 = [[None for a in range(0, num_dim_clusters)] for b in range(num_hands)]
    # merged layers for each hand
    hand_layers = [None for a in range(num_hands)]
    # second conv layer on concatenated conv1 for each hand
    conv2 = [None for a in range(num_hands)]
    # loop over each hand
    for i in range(0, num_hands):
        # loop for each hand over the dimension (or channels) clusters
        for j in range(0, num_dim_clusters):
            # input layer for each dimension cluster for each hand
            x[i][j] = keras.layers.Input(input_shapes[i][j])
            # first conv layer over the clustered dimensions or channels in terms of keras
            conv1[i][j] = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same',
                                              activity_regularizer=regularizers.l2(reg))(x[i][j])
            conv1[i][j] = keras.layers.Activation('relu')(conv1[i][j])
        # concatenate convolutions of first layer over the channels dimension for each hand
        hand_layers[i] = keras.layers.Concatenate(axis=-1)(conv1[i])
        # do a second convolution over features extracted from the first convolution over each hand
        conv2[i] = keras.layers.Conv1D(filters=2 * filters, kernel_size=kernel_size, strides=1, padding='same',
                                       activity_regularizer=regularizers.l2(reg))(hand_layers[i])
        conv2[i] = keras.layers.Activation('relu')(conv2[i])
    # concatenate the features of the two hands
    final_input = keras.layers.Concatenate(axis=-1)(conv2)
    # do a final convolution over the features concatenated for all hands
    conv3 = keras.layers.Conv1D(filters=4 * filters, kernel_size=kernel_size, strides=1, padding='same',
                                activity_regularizer=regularizers.l2(reg))(final_input)
    conv3 = keras.layers.Activation('relu', name="conv_final")(conv3)
    # do a globla average pooling of the final convolution
    pooled = keras.layers.GlobalAveragePooling1D()(conv3)
    # add the final softmax classifier layer
    out = keras.layers.Dense(nb_classes, activation='softmax')(pooled)
    # create the model and link input to output
    model = Model(inputs=list(chain.from_iterable(x)), outputs=out)
    # show summary if specified
    if summary == True:
        model.summary()

    # choose the optimizer
    optimizer = keras.optimizers.Adam(lr=lr, amsgrad=amsgrad)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def results(validation='SuperTrialOut'):
    # s_types = ['Suturing', 'Knot_Tying', 'Needle_Passing']
    s_types = ['Suturing']

    metrics = ['micro', 'macro']

    columns = ['surgery_type', 'validation',
               'cross_val', 'itr', 'architecture',
               'reg', 'lr', 'filters', 'kernel_size',
               'amsgrad', 'val_loss', metrics[0], metrics[1]]

    df_res = pd.DataFrame(index=[], columns=columns)

    curr_df = pd.DataFrame(index=[0], columns=columns)

    curr_df['validation'] = validation

    # for surg in surgery_types:
    for surg in s_types:
        curr_df['surgery_type'] = surg
        dir_out_surgery = out_root_dir + 'JIGSAWS/Experimental_setup/' + surg \
                          + '/unBalanced/GestureClassification/' + validation + '/'

        cross_vals = os.listdir(dir_out_surgery)

        for cross_val in cross_vals:
            curr_df['cross_val'] = cross_val
            if os.path.isdir(dir_out_surgery + cross_val):
                itrs = os.listdir(dir_out_surgery + cross_val)
                for itr in itrs:
                    if os.path.isdir(dir_out_surgery + cross_val + '/' + itr):
                        curr_df['itr'] = itr

                        curr_dir = dir_out_surgery + cross_val + '/' + itr + '/'

                        for subdir, dirs, files in os.walk(curr_dir):
                            if len(files) > 0:

                                file_name = subdir + '/df_metrics.csv'

                                if os.path.exists(file_name):
                                    df_metrics = pd.read_csv(file_name, index_col=False)

                                    arr = subdir.split('/')

                                    curr_df['architecture'] = arr[11].split('__')[1]
                                    curr_df['reg'] = arr[12].split('__')[1]
                                    curr_df['lr'] = arr[13].split('__')[1]
                                    curr_df['filters'] = arr[14].split('__')[1]
                                    curr_df['kernel_size'] = arr[15].split('__')[1]
                                    curr_df['amsgrad'] = arr[16].split('__')[1]

                                    curr_df[metrics[0]] = df_metrics[metrics[0]]
                                    curr_df[metrics[1]] = df_metrics[metrics[1]]
                                    curr_df['val_loss'] = df_metrics['val_loss']

                                    df_res = pd.concat([df_res, curr_df])

    df_res.to_csv(out_root_dir + validation + '-results.csv')


# time
start_time = time.time()

# Global parameters
root_dir = '../JIGSAWS/'
path_to_configurations = '../JIGSAWS/Experimental_setup/'
out_root_dir = '../results/classification/'
nb_epochs = 1000
max_iterations = 1
dimensions_to_use = range(0, 76)
mapSurgeryDataBySurgeryName = collections.OrderedDict()  # indexes surgery data (76 dimensions) by surgery name
mapExpertiseLevelBySurgeryName = collections.OrderedDict()  # indexes exerptise level by surgery name
mapGesturesBySurgeryName = collections.OrderedDict()  # indexes gestures of a surgery by its name
input_shapes = [[(None, 3), (None, 9), (None, 3), (None, 3), (None, 1)],
                [(None, 3), (None, 9), (None, 3), (None, 3), (None, 1)],
                [(None, 3), (None, 9), (None, 3), (None, 3), (None, 1)],
                [(None, 3), (None, 9), (None, 3), (None, 3), (None, 1)]]
# surgery_types = ['Suturing','Knot_Tying','Needle_Passing']
surgery_types = ['Suturing']

if (len(sys.argv) > 1):
    if (sys.argv[1] == 'results'):
        results()
    elif sys.argv[1] == 'cas':
        surgery_type = surgery_types[0]
        surgeries_metadata = getMetaDataForSurgeries(surgery_type)
        cas()
else:
    random.shuffle(surgery_types)

    for surgery_type in surgery_types:
        number_of_dimensions = len(dimensions_to_use)
        input_shape = (
            None,
            number_of_dimensions)  # input is used to specify the value of the second dimension (number of variables)

        # for each hand   x,y,z  ,rot matrx, x'y'z' , a'b'g' , angle  , ... same for the second hand

        #### hyperparam
        architectures = ['fcn']
        regs = [0.00001]
        filterss = [8]
        kernel_sizes = [3]
        lrs = [0.001]
        amsgrads = [0]
        ###############

        classes = ['N', 'I', 'E']
        nb_classes = len(classes)
        confusion_matrix = pd.DataFrame(np.zeros(shape=(nb_classes, nb_classes)), index=classes,
                                        columns=classes)  # matrix used to calculate the JIGSAWS evaluation
        encoder = LabelEncoder()  # used to transform labels into binary one hot vectors

        surgeries_metadata = getMetaDataForSurgeries(surgery_type)

        generateMaps(surgery_type)

        validation(surgery_type, balanced='unBalanced', validation='SuperTrialOut',
                   levelClassify=True, shuff=True)

        print("End!")
