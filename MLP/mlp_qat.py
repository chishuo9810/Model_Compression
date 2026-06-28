import tensorflow as tf 
import os
from tensorflow.keras import layers, models 
from DataGenerator import training_data
# from tensorflow.keras.optimizers import Adam
import datetime
from tensorflow.keras.optimizers.experimental import AdamW
# from tensorflow.keras.optimizers import AdamW 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model
import pdb
EPOCH = 100 # 200
BATCH_SIZE = 256 
FEATURES_BASE = "full"
# MODEL_PATH = "./vitis_pruning/test_0.5_pruning.h5"
MODEL_PATH = "MLP_pruning.h5"
MODEL = 'MLP_' + FEATURES_BASE 
DATE = datetime.date.today()
# full  78
# ga    43
# de 
# pca
tf.config.run_functions_eagerly(True)
from tensorflow_model_optimization.quantization.keras import vitis_quantize
model = load_model(MODEL_PATH)
# pdb.set_trace()
quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy='pof2s')

qat_model = quantizer.get_qat_model()

train_generator, test_generator = training_data(type=FEATURES_BASE)
dim = train_generator[0][0].shape[1] #(BATCH_SIZE, 'DIM', 1)
input_shape = (78,1,1)
num_classes = 15
print("Start Quantize aware Training")
#qat_model.compile(optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=1e-5, weight_decay=1e-9),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
qat_model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-9),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#qat_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-9),
#               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_freq):
        super().__init__()
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            print("")
            print("--------------------------------------------------------------------------------------")
            filepath = f"{MODEL}_quantized_epoch_{epoch + 1:02d}_val_loss_{logs['val_loss']:.4f}_val_accuracy_{logs['val_accuracy']:.4f}.h5"
            # filepath = "test.h5"
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
            print("--------------------------------------------------------------------------------------")
            print("")
            
class BestModelLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch + 1
            print("")
            print("--------------------------------------------------------------------------------------")
            print(f"New best model found at epoch {self.best_epoch} with val_loss {self.best_val_loss:.4f}")
            print("--------------------------------------------------------------------------------------")
            print("")
            
best_model_logger = BestModelLogger()            


            
custom_saver = CustomSaver(save_freq=10)

checkpoint = ModelCheckpoint(f"{MODEL}_best_quantized_{DATE}.h5", save_best_only=True, monitor='val_loss', mode='min') 

early_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min') 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3) 
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, \
                                   write_graph=True, write_images=True) 

# Then run the training process with this qat_model to get the quantize finetuned model.
'''
history = qat_model.fit(train_generator, validation_data=test_generator,  
                    epochs=EPOCH, batch_size=BATCH_SIZE,  
                    callbacks=[checkpoint, early_stop, reduce_lr, tensorboard_callback, custom_saver, best_model_logger]) 
'''
pdb.set_trace()
history = qat_model.fit(train_generator, validation_data=test_generator,
                    epochs=EPOCH, batch_size=BATCH_SIZE,
                    callbacks=[checkpoint, early_stop, reduce_lr, tensorboard_callback, best_model_logger])
print("Loss history:", history.history["loss"])
print("Accuracy history:", history.history["accuracy"])
print("V Loss history:", history.history["val_loss"])
print("V Accuracy history:", history.history["val_accuracy"])
with vitis_quantize.quantize_scope():
    # Save model
    best_path = f"{MODEL}_best_quantized_{DATE}.h5"
    if os.path.exists(best_path):
        qat_model.load_weights(best_path)
        print("Loaded best QAT weights.")
    qat_model.save(f"./vitis_quantize/{MODEL}_full_quantized_{DATE}.h5")
    deploy_quantizer = vitis_quantize.VitisQuantizer(qat_model)
    deploy_model = deploy_quantizer.get_deploy_model(qat_model)
    deploy_model.save(f"./vitis_quantize/{MODEL}_deploy_{DATE}.h5")
    print(f"Deploy model saved to ./vitis_quantize/{MODEL}_deploy_{DATE}.h5")
# qat_model.summary()

print("Done training")
