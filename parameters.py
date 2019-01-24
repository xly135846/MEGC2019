import tensorflow as tf
num_class = 3
width = 28
height = 28
depth = 3
lr = 0.00005
batch_size = 100
epochs = 100
# if total subjects is A, then validation=A*validate_partion
validate_partion = 0.2

image_file_path = '3class_onset_random.txt'
flow_image_path = '/home/zl/Data/optical_flow/data/flow_data_random/'
apex_image_path = '/home/zl/Data/MEdataset/megc2019/3class_onset_random/'
checkpoint_file = 'checkpoint/model'
log_path = 'log'
lamda_x = 0.1

lamda = 1 - lamda_x
is_use_apex = False

tf.app.flags.DEFINE_boolean('is_use_apex', False, '''using original apqx frame in the cnn model''')
# tf.app.flags.DEFINE_boolean('is_use_apex', False, '''using original apqx frame in the cnn model''')