from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.optimizers import *
from image_process import *
from parameters import *
from callbacks import *
from keras import backend as K
import tensorflow as tf
import pandas as pd

from sklearn.metrics import f1_score, confusion_matrix

train_num_list = []
full_f1_list = []
casme2_f1_list = []
samm_f1_list = []
smic_f1_list = []

full_uar_list = []
casme2_uar_list = []
samm_uar_list = []
smic_uar_list = []

full_matrix_list = []
casme2_matrix_list = []
samm_matrix_list = []
smic_matrix_list = []

def evaluate_metrics(predict_label, true_label, num):
    # predict_label = np.array([0, 1, 2])
    # true_label = np.array([0, 2, 1])
    # macro_f1

    full_f1 = f1_score(true_label, predict_label, average='macro')
    casme2_f1 = f1_score(true_label[0:145], predict_label[0:145], average='macro')
    samm_f1 = f1_score(true_label[145:278], predict_label[145:278], average='macro')
    smic_f1 = f1_score(true_label[278:], predict_label[278:], average='macro')

    full_f1_list.append(full_f1)
    casme2_f1_list.append(casme2_f1)
    samm_f1_list.append(samm_f1)
    smic_f1_list.append(smic_f1)

    print('f1 macro score:', full_f1)
    print('CASME2 f1 macro score:', casme2_f1)
    print('SAMM f1 macro score:', samm_f1)
    print('SMIC-hs f1 macro score:', smic_f1)

    # matrix
    confusion_matrixs = confusion_matrix(true_label, predict_label)
    casme2_confusion_matrix = confusion_matrix(true_label[0:145], predict_label[0:145])
    samm_confusion_matrix = confusion_matrix(true_label[145:278], predict_label[145:278])
    smic_hs_confusion_matrix = confusion_matrix(true_label[278:], predict_label[278:])

    full_matrix_list.append(confusion_matrixs)
    casme2_matrix_list.append(casme2_confusion_matrix)
    samm_matrix_list.append(samm_confusion_matrix)
    smic_matrix_list.append(smic_hs_confusion_matrix)
    print('full confusion_matrix:', confusion_matrixs)
    print('casme2 confusion_matrix:', casme2_confusion_matrix)
    print('samm confusion_matrix:', samm_confusion_matrix)
    print('smic_hs confusion_matrix:', smic_hs_confusion_matrix)

    #uar
    # total uar
    d = np.diag(confusion_matrixs)
    m = np.sum(confusion_matrixs, axis=1)
    uar = np.sum(d/m/num_class)
    print('uar:', uar)
    # casme2 uar
    casme2_d = np.diag(casme2_confusion_matrix)
    casme2_m = np.sum(casme2_confusion_matrix, axis=1)
    casme2_uar = np.sum(casme2_d/casme2_m/num_class)
    print('casme2_uar:', casme2_uar)
    # samm uar
    samm_d = np.diag(samm_confusion_matrix)
    samm_m = np.sum(samm_confusion_matrix, axis=1)
    samm_uar = np.sum(samm_d / samm_m / num_class)
    print('samm_uar:', samm_uar)
    #hs uar
    smic_hs_d = np.diag(smic_hs_confusion_matrix)
    smic_hs_m = np.sum(smic_hs_confusion_matrix, axis=1)
    smic_hs_uar = np.sum(smic_hs_d/smic_hs_m/num_class)
    print('smic_hs_uar:', smic_hs_uar)

    full_uar_list.append(uar)
    casme2_uar_list.append(casme2_uar)
    samm_uar_list.append(samm_uar)
    smic_uar_list.append(smic_hs_uar)

    train_num_list.append(num)
    df = pd.DataFrame(data={'train_num': train_num_list,
                            'full_f1_list': full_f1_list, 'casme2_f1_list': casme2_f1_list,
                            'samm_f1_list':samm_f1_list, 'smic_f1_list': smic_f1_list,
                            'full_uar_list': full_uar_list, 'casme2_uar_list': casme2_uar_list,
                            'samm_uar_list': samm_uar_list, 'smic_uar_list': smic_uar_list,
                            'full_matrix_list': full_matrix_list, 'casme2_matrix_list': casme2_matrix_list,
                            'samm_matrix_list': samm_matrix_list, 'smic_matrix_list': smic_matrix_list
                            })
    df.to_csv('save_result/metrics.csv')

def create_inception_model():

    input_1 = Input(shape=[28,28,3],name='inputs_x')
    input_2 = Input(shape=[28,28,3],name='inputs_y')


    ince_conv1_1 = Conv2D(6, (1,1), strides=(1,1), activation='relu', padding='same')(input_1)

    ince_conv1_2 = Conv2D(6, (1, 1), strides=(1, 1), activation='relu', padding='same')(input_1)
    ince_conv1_2 = Conv2D(6, (3, 3), strides=(1, 1), activation='relu', padding='same')(ince_conv1_2)

    ince_conv1_3 = Conv2D(6, (1, 1), strides=(1, 1), activation='relu', padding='same')(input_1)
    ince_conv1_3 = Conv2D(6, (5, 5), strides=(1, 1), activation='relu', padding='same')(ince_conv1_3)

    ince_1_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_1)
    ince_1_4 = Conv2D(6, (1, 1), strides=(1, 1), activation='relu', padding='same')(ince_1_4)

    ince_con_1 = Concatenate()([ince_conv1_1, ince_conv1_2, ince_conv1_3, ince_1_4])

    ince_pooling_1 = MaxPooling2D((2,2))(ince_con_1)

    ince_conv2_1 = Conv2D(16, (1,1), strides=(1,1), activation='relu', padding='same')(ince_pooling_1)

    ince_conv2_2 = Conv2D(16, (1, 1), strides=(1, 1), activation='relu', padding='same')(ince_pooling_1)
    ince_conv2_2 = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same')(ince_conv2_2)

    ince_conv2_3 = Conv2D(16, (1, 1), strides=(1, 1), activation='relu', padding='same')(ince_pooling_1)
    ince_conv2_3 = Conv2D(16, (5, 5), strides=(1, 1), activation='relu', padding='same')(ince_conv2_3)
    ince_2_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(ince_pooling_1)
    ince_2_4 = Conv2D(16, (1, 1), strides=(1, 1), activation='relu', padding='same')(ince_2_4)

    ince_con_2 = Concatenate()([ince_conv2_1, ince_conv2_2, ince_conv2_3, ince_2_4])

    ince_pooling_2_x = MaxPooling2D((2, 2))(ince_con_2)


    ince_conv1_1 = Conv2D(6, (1,1), strides=(1,1), activation='relu', padding='same')(input_2)

    ince_conv1_2 = Conv2D(6, (1, 1), strides=(1, 1), activation='relu', padding='same')(input_2)
    ince_conv1_2 = Conv2D(6, (3, 3), strides=(1, 1), activation='relu', padding='same')(ince_conv1_2)

    ince_conv1_3 = Conv2D(6, (1, 1), strides=(1, 1), activation='relu', padding='same')(input_2)
    ince_conv1_3 = Conv2D(6, (5, 5), strides=(1, 1), activation='relu', padding='same')(ince_conv1_3)

    ince_1_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_2)
    ince_1_4 = Conv2D(6, (1, 1), strides=(1, 1), activation='relu', padding='same')(ince_1_4)

    ince_con_1 = Concatenate()([ince_conv1_1, ince_conv1_2, ince_conv1_3, ince_1_4])

    ince_pooling_1 = MaxPooling2D((2,2))(ince_con_1)

    ince_conv2_1 = Conv2D(16, (1,1), strides=(1,1), activation='relu', padding='same')(ince_pooling_1)

    ince_conv2_2 = Conv2D(16, (1, 1), strides=(1, 1), activation='relu', padding='same')(ince_pooling_1)
    ince_conv2_2 = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same')(ince_conv2_2)

    ince_conv2_3 = Conv2D(16, (1, 1), strides=(1, 1), activation='relu', padding='same')(ince_pooling_1)
    ince_conv2_3 = Conv2D(16, (5, 5), strides=(1, 1), activation='relu', padding='same')(ince_conv2_3)

    ince_2_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(ince_pooling_1)
    ince_2_4 = Conv2D(16, (1, 1), strides=(1, 1), activation='relu', padding='same')(ince_2_4)

    ince_con_2 = Concatenate()([ince_conv2_1, ince_conv2_2, ince_conv2_3, ince_2_4])

    ince_pooling_2_y = MaxPooling2D((2, 2))(ince_con_2)

    incepf_1 = Flatten()(ince_pooling_2_x)
    incepf_2 = Flatten()(ince_pooling_2_y)

    x = Concatenate()([incepf_1,incepf_2])

    x = Dense(1024,activation='relu')(x)
    # x = Dense(1024,activation='relu')(x)
    x = Dense(num_class,activation='softmax')(x)

    model = Model(inputs=[input_1,input_2], outputs=x)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # compile model
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':

    # model = create_inception_model()
    # # model.summary()
    # plot_model(model, model.name+'.png', show_shapes=True, show_layer_names=True)
    [filename, train_sub, val_sub, test_sub] = create_file_list()
    for round_time in range(100):
        numth_test_sample_list = []
        predict_label_list = []
        true_label_list = []
        for sub in range(68):
            print('********  ', sub, '   folder start:', test_sub[sub])
            model = create_inception_model()
            [train_x_array, train_y_array, train_label,
             val_x_array, val_y_array, val_label,
             test_x_array, test_y_array, test_label, x_list] = read_file_directly(filename, sub, train_sub, val_sub, test_sub)
            for len in range(np.shape(x_list)[0]):
                numth_test_sample_list.append(x_list[len])
            train_label = to_categorical(train_label, num_class)
            test_label = to_categorical(test_label, num_class)
            val_label = to_categorical(val_label, num_class)

            # checkpoint_path = 'checkpoint/model'+str(sub)+'.h5'
            checkpoint_path = 'checkpoint/model.h5'

            checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1)
            callback_list = [checkpoint, tensorborad]

            hist = model.fit({'inputs_x':train_x_array, 'inputs_y':train_y_array}, train_label,
                             validation_data=({'inputs_x':val_x_array,'inputs_y':val_y_array}, val_label),
                             class_weight='auto',
                             shuffle=True,
                             batch_size=3, epochs=40, verbose=0, callbacks=callback_list)  # starts training
            # save loss
            his_log = 'his_log/inception' + str(round_time) + '_' + str(sub) + '.txt'
            with open(his_log, 'w') as f:
                f.write(str(hist.history))
                f.close()
            model2 = create_inception_model()
            model2.load_weights(checkpoint_path)
            emotion_classes = model2.predict({'inputs_x': test_x_array, 'inputs_y': test_y_array}, batch_size=1)
            # revert clssed to 0,1,2...,and save to predict file
            predict_label = np.argmax(emotion_classes, axis=1)
            predict_label_list += list(predict_label)
            true_label = np.argmax(test_label, axis=1)
            true_label_list += list(true_label)
            K.clear_session()
            tf.reset_default_graph()
        # save_predict_label
        predict_label_list = np.array(predict_label_list)
        result_f = open('save_result/result_' + str(round_time)+'.txt', 'w')
        result_f.write(str(predict_label_list))
        result_f.close()
        #save_true label
        true_f = open('save_result/true_' + str(round_time)+'.txt', 'w')
        true_label_list = np.array(true_label_list)
        true_f.write(str(true_label_list))
        true_f.close()
        # save file names
        file_f = open('save_result/file_names' + str(round_time) + '.txt', 'w')
        # numth_test_sample_list = np.ndarray(numth_test_sample_list)
        numth_test_sample_list = np.array(numth_test_sample_list)

        # print(numth_test_sample_list)
        file_f.write(str(numth_test_sample_list))
        file_f.close()
        evaluate_metrics(predict_label_list, true_label_list, round_time)

        numth_test_sample_list = numth_test_sample_list.tolist()
        true_label_list = true_label_list.tolist()
        predict_label_list = predict_label_list.tolist()

        da = pd.DataFrame(data={'file_list': numth_test_sample_list, 'true_label': true_label_list,
                                'predict_label': predict_label_list
                                })
        da.to_csv('save_result/results_' + str(round_time) + '.csv')

        # print the evaluate results






