from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Input, Dense, MaxPooling2D, Softmax, Add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from DataGenerator import training_data
from tensorflow.keras.regularizers import l2

# Create mlp model
def create_mlp_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)

    units = [512, 256, 256, 128, 128, 64, 64, 32]

    for i, u in enumerate(units):
        x = Dense(
            u,
            kernel_regularizer=l2(0.01),
            name=f"dense_{i}"
        )(x)
        x = ReLU(max_value=6, name=f"relu_{i}")(x)

    x = Dense(num_classes, name="classifier")(x)
    outputs = Softmax(name="softmax")(x)

    model = Model(inputs, outputs)
    return model

EPOCH = 50 
BATCH_SIZE = 1024 
FEATURES_BASE = "full"
Kernrel_Regular = "001"
MODEL = 'MLP_10layers_' + Kernrel_Regular + FEATURES_BASE
# full  78
# ga    43
# de 
# pca
train_generator, test_generator = training_data(type=FEATURES_BASE)
dim = train_generator[0][0].shape[1] #(BATCH_SIZE, 'DIM', 1)
input_shape = (dim,1,1)
num_classes = 15 

model = create_mlp_model(input_shape, num_classes)
model.summary()

'''print('Inspect...')
from tensorflow_model_optimization.quantization.keras import vitis_inspect
inspector = vitis_inspect.VitisInspector(target="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json")
filename_dump = f"{MODEL}.txt"
filename_svg  = f"{MODEL}.svg"
inspector.inspect_model(model,
                        input_shape=[1, *input_shape],
                        plot=True,
                        plot_file=filename_svg,
                        dump_results=True,
                        dump_results_file=filename_dump,
                        verbose=0)'''

model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Model checkpoint and early stopping
checkpoint = ModelCheckpoint(f"./mlp_model/{MODEL}_best.h5", save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, \
                                   write_graph=True, write_images=True) 

print("Start to train model")
# Train on data
history = model.fit(train_generator, validation_data=test_generator, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[checkpoint, early_stop, reduce_lr])

print("Loss history:", history.history["loss"])
print("Accuracy history:", history.history["accuracy"])
print("V Loss history:", history.history["val_loss"])
print("V Accuracy history:", history.history["val_accuracy"])

# Save model
model.save(f"./mlp_model/{MODEL}_final.h5")
print("Done training")
