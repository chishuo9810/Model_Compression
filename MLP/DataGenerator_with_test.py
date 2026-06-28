# 07/23 04:30 version 
# Intergrate all type within a function

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split

BATCH_SIZE = 1024
TRAIN_DATA_PATH = '../meta/balanced_data.csv'
SCALER = '../meta/scaler.pkl'
ENCODER = '../meta/label_enc.pkl'
default_to_drop = ['Label', 'Timestamp', 'Src IP', 'Src Port', 'Flow ID', 'Dst IP']
# if len was wrong, check Label. There is overlap
# Total columns = 79
ga_features_to_drop = [         # len() = 36
    'Protocol',
    'Tot Fwd Pkts',
    'Tot Bwd Pkts',
    'Fwd Pkt Len Min',
    'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean',
    'Bwd Pkt Len Std',
    'Fwd IAT Tot',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Bwd Pkts/s',
    'Pkt Len Min',
    'Pkt Len Std',
    'FIN Flag Cnt',
    'SYN Flag Cnt',
    'ACK Flag Cnt',
    'URG Flag Cnt',
    'CWE Flag Count',
    'Pkt Size Avg',
    'Fwd Seg Size Avg',
    'Fwd Byts/b Avg',
    'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg',
    'Bwd Pkts/b Avg',
    'Bwd Blk Rate Avg',
    'Subflow Fwd Pkts',
    'Subflow Bwd Pkts',
    'Fwd Act Data Pkts',
    'Active Mean',
    'Active Std',
    'Active Max',
    'Idle Max',
    'Idle Min',
    'Label'
]
de_features_to_drop = [         # len() = 
]
pca_features_to_drop = [        # len() = 

]
scaler = joblib.load(SCALER)
label_enc = joblib.load(ENCODER)

# DataGenerator Class
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        return batch_data, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def training_data(dataset=TRAIN_DATA_PATH, type="full"):  
    print(f"Read dataset {dataset}") 
    df = pd.read_csv(dataset) 

    features = df.drop(columns=default_to_drop, errors='ignore') 
    features_scaled = scaler.transform(features) 

    if(type == 'full'):
        input_shape = (len(features.columns), 1, 1)  # (78,1,1) 
        features_using = features_scaled
    elif(type == 'ga'):
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
        features_scaled_reduced_dim_ga = \
            features_scaled_df.drop(columns=ga_features_to_drop, errors='ignore')
        input_shape = (len(features_scaled_reduced_dim_ga.columns), 1, 1)  # (43,1,1)
        features_using = features_scaled_reduced_dim_ga
    elif(type == 'de'):
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
        features_scaled_reduced_dim_de = \
            features_scaled_df.drop(columns=de_features_to_drop, errors='ignore')
        input_shape = (len(features_scaled_reduced_dim_de.columns), 1, 1)  # (?,1,1)
        features_using = features_scaled_reduced_dim_de
    elif(type == 'pca'):
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
        features_scaled_reduced_dim_pca = \
            features_scaled_df.drop(columns=pca_features_to_drop, errors='ignore')
        input_shape = (len(features_scaled_reduced_dim_pca.columns), 1, 1)  # (?,1,1)
        features_using = features_scaled_reduced_dim_pca
    else:
        raise ValueError("Feature type is not allowed") 
    
    num_classes = len(label_enc.classes_) 
    print("Input shape:", input_shape) 
    print("# of Class:", num_classes) 
 
    print("Norm and encode") 
    expanded_data = np.expand_dims(features_using, axis=-1) 
    labels = df['Label']
    labels_enc = label_enc.transform(labels) 
    
    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        expanded_data, labels_enc,
        test_size=0.2,
        random_state=42,
        stratify=labels_enc
    )

    X_test, X_val, Y_test, Y_val = train_test_split(
        X_tmp, Y_tmp,
        test_size=0.5,
        random_state=42,
        stratify=Y_tmp
    )

    train_generator = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_generator   = DataGenerator(X_val,   Y_val,   batch_size=BATCH_SIZE, shuffle=False)
    test_generator  = DataGenerator(X_test,  Y_test,  batch_size=BATCH_SIZE, shuffle=False)

    return train_generator, val_generator, test_generator
