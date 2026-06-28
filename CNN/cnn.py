import tensorflow as tf 
from tensorflow.keras import layers, models 
from DataGenerator import training_data
from tensorflow.keras.optimizers import AdamW 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
def cnn(input_shape, num_classes): 
    inputs = layers.Input(shape=input_shape) 
    x = inputs 
    x = layers.Conv2D(16, (3, 1), kernel_regularizer=l2(0.0112))(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.Conv2D(16, (3, 1), kernel_regularizer=l2(0.0112))(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.BatchNormalization()(x) 
 
    x = layers.Conv2D(32, (5, 1), kernel_regularizer=l2(0.0112))(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.MaxPooling2D((2,1))(x) 
    x = layers.BatchNormalization()(x) 
    x = layers.Flatten()(x) 
    x = layers.Dense(64)(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.Dense(num_classes)(x) 
     
    outputs = layers.Softmax()(x) 
    model = models.Model(inputs, outputs) 
    return model 

EPOCH = 50 
BATCH_SIZE = 64 
FEATURES_BASE = "full"
MODEL = 'CNN_00112_full_without_denseRegular' + FEATURES_BASE
# full  78
# ga    43
# de 
# pca
train_generator, test_generator = training_data(type=FEATURES_BASE)
dim = train_generator[0][0].shape[1] #(BATCH_SIZE, 'DIM', 1)
input_shape = (dim,1,1)
num_classes = 15 

DEBUG = 1
if DEBUG:
    print("Input shape: ", input_shape)

model = cnn(input_shape, num_classes) 
model.summary()
model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5), 
                loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
 
# Model checkpoint and early stopping 
checkpoint = ModelCheckpoint(f"./cnn_model/{MODEL}_best.h5", save_best_only=True, monitor='val_loss', mode='min') 
early_stop = EarlyStopping(monitor='val_loss', patience=8, mode='min') 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001) 
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, \
                                   write_graph=True, write_images=True) 
  
print("Start to train model") 
history = model.fit(train_generator, validation_data=test_generator,  
                    epochs=EPOCH, batch_size=BATCH_SIZE,  
                    callbacks=[checkpoint, early_stop, reduce_lr, tensorboard_callback]) 
 
print("Loss history:", history.history["loss"]) 
print("Accuracy history:", history.history["accuracy"]) 
print("V Loss history:", history.history["val_loss"]) 
print("V Accuracy history:", history.history["val_accuracy"]) 
 
# Save model 
model.save(f"./cnn_model/{MODEL}_final.h5") 
print("Done training")
'''
"""
bug existing 
"""
import numpy as np
XMODEL_PATH = ''
batch_data = np.array()
input_ndim, output_ndim = ()
import vart
import xir
graph = xir.Graph.deserialize(XMODEL_PATH)
subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
dpu_subgraph = [sg for sg in subgraphs if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"]
dpu_subgraph = dpu_subgraph[0]

runner = vart.Runner.create_runner(dpu_subgraph, 'run')
print("Initiated DPU runner")

input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()

input_data_buf = batch_data.reshape(input_ndim).astype(np.float32)
output_data_buf = np.zeros(output_ndim, dtype=np.float32)
job_id = runner.execute_async([input_data_buf], [output_data_buf])
'''
