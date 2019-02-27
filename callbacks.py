from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from parameters import *


# checkpoint = ModelCheckpoint(filepath=checkpoint_file, save_weights_only=True, save_best_only=True, verbose=2)
tensorborad = TensorBoard(log_dir=log_path)
early_stop = EarlyStopping(patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_', patience=5, verbose=2, factor=0.99, min_lr=0.00001)

# callback_list = [checkpoint, tensorborad, early_stop]
