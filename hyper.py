import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from enum import Enum

class CaptchaType(Enum):
    SUPREME_COURT = "supreme_court"
    GOV24 = "gov24"
    NH_WEB_MAIL = "nh_web_mail" 

class Hyper:
    
    def __init__(self):
        self.NULL_OUT = open(os.devnull, 'w')
        self.STD_OUT = sys.stdout

    def quiet(self, value:bool):

        import absl.logging
        
        if value:
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            absl.logging.set_verbosity(absl.logging.ERROR)
            sys.stdout = self.NULL_OUT
        else:
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            import tensorflow as tf
            tf.get_logger().setLevel('INFO')
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
            absl.logging.set_verbosity(absl.logging.INFO)
            sys.stdout = self.STD_OUT

    def get_base_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def get_image_files(self, cap_type:CaptchaType, train=True):
        baseDir = self.get_base_dir()
        imgDir = os.path.join(baseDir, "images", cap_type.value, "train" if train else "pred")
        return Path(imgDir).glob("*.png")

    def get_image_info(self, img_path_list:list):
        img_path = img_path_list[-1]
        img = Image.open(img_path)
        img_width = img.width
        img_height = img.height
        return img_width, img_height

    def get_train_info(self, train_img_path_list):
        labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in train_img_path_list]
        max_length = max([len(label) for label in labels])
        characters = sorted(set(char for label in labels for char in label))
        return max_length, characters

    def get_weights_path(self, cap_type:CaptchaType, weightsOnly=True):
        baseDir = self.get_base_dir()
        weightsDir = os.path.join(baseDir, "model", cap_type.value)
        if weightsOnly:
            weightsDir = weightsDir + ".weights.h5"

        return weightsDir

    def get_model(self, captcha_type:CaptchaType, weight_only):
        train_img_path_list = self.get_image_files(captcha_type, train=True)
        img_width, img_height = self.get_image_info(train_img_path_list)
        max_length, characters = self.get_train_info(train_img_path_list)
        weights_path = self.get_weights_path(captcha_type, weight_only)
        model = ApplyModel(weights_path, img_width, img_height, max_length, characters)
        return model

    def model_train(self, captcha_type:CaptchaType, patience:int=7):
        train_img_path_list = self.get_image_files(captcha_type, train=True)
        img_width, img_height = self.get_image_info(train_img_path_list)
        CM = CreateModel(train_img_path_list, img_width, img_height)
        model = CM.train_model(epochs=100, earlystopping=True, early_stopping_patience=patience)
        weights_path = self.get_weights_path(captcha_type, True)
        model.save_weights(weights_path)
        weights_path = self.get_weights_path(captcha_type, False)
        model.save(weights_path)

    def model_validate(self, captcha_type:CaptchaType, weight_only=True):
        start = time.time()
        matched = 0

        pred_img_path_list = self.get_image_files(captcha_type, train=False)
        train_img_path_list = self.get_image_files(captcha_type, train=True)
        img_width, img_height = self.get_image_info(train_img_path_list)
        max_length, characters = self.get_train_info(train_img_path_list)
        weights_path = self.get_weights_path(captcha_type, weight_only)
        model = ApplyModel(weights_path, img_width, img_height, max_length, characters)

        for pred_img_path in pred_img_path_list:
            pred = model.predict(pred_img_path)
            ori = pred_img_path.split(os.path.sep)[-1].split(".")[0]
            msg = ""
            if(ori == pred):
                matched += 1
            else:
                msg = " Not matched!"
            print("ori : ", ori, "pred : ", pred, msg)

        end = time.time()

        print("Matched:", matched, ", Tottal : ", len(pred_img_path_list))
        print("pred time : ", end - start, "sec")

    def predict(self, captcha_type:CaptchaType, weight_only, image_path:str):
        self.quiet(True)
        model = self.get_model(captcha_type, weight_only)
        pred = model.predict(image_path)
        return pred

    def predict_from_bytes(self, captcha_type:CaptchaType, weight_only, image_bytes:bytes):
        self.quiet(True)
        model = self.get_model(captcha_type, weight_only)
        pred = model.predict_from_bytes(image_bytes)
        return pred

import tensorflow as tf
import keras
from keras import ops
from keras import layers

characters = ['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']

# Mapping characters to integers
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

# Splitting data into training and validation sets
data_dir = Path("./images/nh_web_mail/train/")
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]

# Batch size for training and validation
batch_size = 16

# Desired image dimensions
img_width = 200
img_height = 50

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4

# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])

def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = ops.arange(size)
    if shuffle:
        keras.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid

x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))

def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = ops.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = ops.transpose(img, axes=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = ops.cast(
        ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
    )

    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

    return ops.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )


def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(
        ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(
        ops.reshape(
            ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(
        ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        ops.cast(indices, dtype="int64"), 
        vals_sparse, 
        ops.cast(label_shape, dtype="int64")
    )


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


# Get the model
model = build_model()
model.summary()

class TrainModel:
    def __init__(self, captcha_type:CaptchaType, weights_only=True):
        self.captcha_type = captcha_type
        self.weights_only = weights_only
        self.hyper = Hyper()
        self.weights_path = self.hyper.get_weights_path(captcha_type, weights_only)
        self.train_img_path_list = self.hyper.get_image_files(captcha_type, train=True)
        self.img_width, self.img_height = self.hyper.get_image_info(self.train_img_path_list)
        self.max_length, self.characters = self.hyper.get_train_info(self.train_img_path_list)
        self.model = CreateModel(self.train_img_path_list, self.img_width, self.img_height)
    

class CreateModel:
    
    def __init__(self, train_img_path, img_width=200, img_height=50):
        # 이미지 크기
        self.img_width = img_width
        self.img_height = img_height
        # 학습 이미지 파일 경로 리스트
        self.images = sorted(train_img_path)
        # 학습 이미지 파일 라벨 리스트
        self.labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in self.images]
        # 라벨 SET
        self.characters = set(char for label in self.labels for char in label)
        # 라벨 최대 길이
        self.max_length = max([len(label) for label in self.labels])

        # Mapping characters to integers
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=sorted(self.characters), num_oov_indices=0, mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        
    def train_model(self, epochs=100, earlystopping=False, early_stopping_patience=5):
        # 학습 및 검증을 위한 배치 사이즈 정의
        batch_size = 16
        # 다운 샘플링 요인 수 (Conv: 2, Pooling: 2)
        downsample_factor = 4
        
        # Splitting data into training and validation sets
        x_train, x_valid, y_train, y_valid = self.split_data(np.array(self.images), np.array(self.labels))

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = (
            validation_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        
        # Get the model
        model = self.build_model()
        
        if earlystopping == True:

            # early_stopping_patience = 5
            # Add early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
            )

            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=[early_stopping],
            )
        
        else:
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs
            )

        
        return model
    
        
    def encode_single_sample(self, img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [self.img_height, self.img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}
    
    def build_model(self):
        # Inputs to the model
        input_img = layers.Input(
            shape=(self.img_width, self.img_height, 1), name="image", dtype="float32"
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # First conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(len(self.characters) + 1, activation="softmax", name="dense2")(x)

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        opt = keras.optimizers.Adam()
        # Compile the model and return
        model.compile(optimizer=opt)
        return model
    
    def split_data(self, images, labels, train_size=0.9, shuffle=True):
        # 1. Get the total size of the dataset
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        return x_train, x_valid, y_train, y_valid

class ApplyModel:
    
    def __init__(self, 
                 weights_path,
                 img_width=200, 
                 img_height=50, 
                 max_length=6, 
                 characters={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}):
        
        self.img_width = img_width
        self.img_height = img_height
        self.max_length = max_length
        self.characters = characters
                
        # Mapping characters to integers
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=sorted(self.characters), num_oov_indices=0, mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        # Model
        self.model = self.build_model()

        if os.path.splitext(weights_path)[-1] == ".h5":
            self.model.load_weights(weights_path)
        else:
            self.model = keras.models.load_model(weights_path)

        self.prediction_model = keras.models.Model(
            self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output
        )
    
    def predict(self, target_img_path):
        target_img = self.encode_single_sample(target_img_path)['image']
        target_img = tf.reshape(target_img, shape=[1,self.img_width,self.img_height,1])
        pred_val = self.prediction_model.predict(target_img)
        pred = self.decode_batch_predictions(pred_val)[0]
        return pred

    def predict_from_bytes(self, image_bytes):
        target_img = self.encode_single_sample_from_bytes(image_bytes)['image']
        target_img = tf.expand_dims(target_img, 0)
        pred_val = self.prediction_model.predict(target_img)
        pred = self.decode_batch_predictions(pred_val)[0]
        return pred

    def encode_single_sample(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        return {"image": img}

    def encode_single_sample_from_bytes(self, image_bytes):
        img = tf.io.decode_image(image_bytes, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        return {"image": img}

    def build_model(self):
        # Inputs to the model
        input_img = layers.Input(
            shape=(self.img_width, self.img_height, 1), name="image", dtype="float32"
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # First conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(len(self.characters) + 1, activation="softmax", name="dense2")(x)

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        opt = keras.optimizers.Adam()
        # Compile the model and return
        model.compile(optimizer=opt)
        return model
    
    def split_data(self, images, labels, train_size=0.9, shuffle=True):
        # 1. Get the total size of the dataset
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        return x_train, x_valid, y_train, y_valid

    # A utility function to decode the output of the network
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.max_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res+1)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text
