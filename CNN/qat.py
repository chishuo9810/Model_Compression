import tensorflow as tf 
from tensorflow.keras import layers, models 
from DataGenerator import training_data
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model
def cnn(input_shape, num_classes): 
    inputs = layers.Input(shape=input_shape) 
    x = inputs 
    x = layers.Conv2D(16, (3, 1))(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.Conv2D(16, (3, 1))(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.BatchNormalization()(x) 
 
    x = layers.Conv2D(32, (3, 1))(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.Conv2D(32, (3, 1))(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.BatchNormalization()(x) 
 
    x = layers.Conv2D(32, (5, 1))(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.MaxPooling2D((2,1))(x) 
    x = layers.BatchNormalization()(x) 
    x = layers.Flatten()(x) 
    x = layers.Dense(128)(x) 
    x = layers.LeakyReLU(alpha=0.1015625)(x) 
    x = layers.Dense(64)(x) 
    x = layers.ReLU(max_value=6)(x) 
    x = layers.Dense(num_classes)(x) 
     
    outputs = layers.Softmax()(x) 
    model = models.Model(inputs, outputs) 
    return model 

EPOCH = 50 
BATCH_SIZE = 128 
FEATURES_BASE = "ga"
MODEL = 'CNN_' + FEATURES_BASE
QUANTIZE_PATH = '../meta/test_data.csv'
# full  78
# ga    43
# de 
# pca
train_generator, test_generator = training_data(type=FEATURES_BASE)
dim = train_generator[0][0].shape[1] #(BATCH_SIZE, 'DIM', 1)
input_shape = (dim,1,1)
num_classes = 15 

'''DEBUG = 1
if DEBUG:
    print("Input shape: ", input_shape)

model = cnn(input_shape, num_classes) 
model.summary()
model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5), 
                loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
 
# Model checkpoint and early stopping 
checkpoint = ModelCheckpoint(f"{MODEL}_best.h5", save_best_only=True, monitor='val_loss', mode='min') 
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
model.save(f"{MODEL}_final.h5") 
print("Done training")'''

model = load_model("CNN_ga_best.h5")
print("Start Quantize aware Training")
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy='pof2s')
qat_model = quantizer.get_qat_model()

qat_model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-8), 
        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#qat_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-8),
#               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(f"{MODEL}_best_quantized.h5", save_best_only=True, monitor='val_loss', mode='min') 
early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min') 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001) 
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, \
                                   write_graph=True, write_images=True) 

# Then run the training process with this qat_model to get the quantize finetuned model.
history = qat_model.fit(train_generator, validation_data=test_generator,  
                    epochs=EPOCH, batch_size=BATCH_SIZE,  
                    callbacks=[checkpoint, early_stop, reduce_lr, tensorboard_callback]) 
print("Loss history:", history.history["loss"])
print("Accuracy history:", history.history["accuracy"])
print("V Loss history:", history.history["val_loss"])
print("V Accuracy history:", history.history["val_accuracy"])
with vitis_quantize.quantize_scope():
    # Save model
    qat_model.save(f"{MODEL}_quantized.h5")
# qat_model.summary()

print("Done training")
