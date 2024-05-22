import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
from enum import Enum
from pathlib import Path
from PIL import Image

class CaptchaType(Enum):
    SUPREME_COURT = "supreme_court"
    GOV24 = "gov24"
    NH_WEB_MAIL = "nh_web_mail" 

class Hyper:

    print("Hyper class declared")

    def __init__(self, captcha_type:CaptchaType=CaptchaType.SUPREME_COURT, weights_only=True, quiet_out=False):
        print("Hyper class init")

        self.NULL_OUT = open(os.devnull, 'w')
        self.STD_OUT = sys.stdout

        self.captcha_type = captcha_type
        self.weights_only = weights_only
        self.quiet_out = quiet_out

        if self.quiet_out:
            self.quiet(True)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_image_paths = self.image_paths(True)
        self.pred_image_paths = self.image_paths(False)
        self.model_path = self.saved_model_path()
        self.image_width, self.image_height, self.max_length, self.characters = self.train_info()

        import keras
        from keras import layers

        # Mapping characters to integers
        self.char_to_num = layers.StringLookup(vocabulary=list(self.characters), mask_token=None)

        # Mapping integers back to original characters
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

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

    def image_paths(self, train=True):
        imgDir = os.path.join(self.base_dir, "images", self.captcha_type.value, "train" if train else "pred")
        return list(Path(imgDir).glob("*.png"))

    def train_info(self):
        image_path = self.train_image_paths[-1]
        image = Image.open(image_path)
        image_width = image.width
        image_height = image.height
        labels = [ os.path.splitext(train_image_file.name)[0] for train_image_file in self.train_image_paths]
        max_length = max([len(label) for label in labels])
        characters = sorted(set(char for label in labels for char in label))
        return image_width, image_height, max_length, characters

    def saved_model_path(self):
        return os.path.join(self.base_dir, "model", self.captcha_type.value, ".weights.h5" if self.weights_only else "")

    def split_data(self):
        
        import numpy as np
        
        images = np.array([train_image_paths.absolute() for train_image_paths in self.train_image_paths])
        labels = np.array([os.path.splitext(train_image_path.name)[0] for train_image_path in self.train_image_paths])
        train_size=0.9
        shuffle=True

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

    def build_model(self):
        import keras
        from keras import layers

        # Inputs to the model
        input_img = layers.Input(
            shape=(self.image_width, self.image_height, 1), name="image", dtype="float32"
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
        new_shape = ((self.image_width // 4), (self.image_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(
            len(self.char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
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
    
    def encode_single_sample(self, img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = ops.image.resize(img, [self.image_height, self.image_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = ops.transpose(img, axes=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}

    def validate_model(self):
        import keras
        import time

        start = time.time()
        matched = 0

        if self.weights_only:
            self.model = self.build_model()
            self.model.load_weights(self.model_path)
        else:
            self.model = keras.models.load_model(self.model_path)

        self.prediction_model = keras.models.Model(
            self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output
        )

        for pred_img_path in self.pred_image_paths:
            pred = self.prediction_model.predict(pred_img_path.absolute())
            ori = pred_img_path.split(os.path.sep)[-1].split(".")[0]
            msg = ""
            if(ori == pred):
                matched += 1
            else:
                msg = " Not matched!"
            print("ori : ", ori, "pred : ", pred, msg)

        end = time.time()
        print("Matched:", matched, ", Tottal : ", len(self.pred_image_paths))
        print("pred time : ", end - start, "sec")

    def train_model(self, epochs=100, earlystopping=True, early_stopping_patience=7):
        
        # import matplotlib.pyplot as plt


        # 학습 및 검증을 위한 배치 사이즈 정의
        batch_size = 16
        # 다운 샘플링 요인 수 (Conv: 2, Pooling: 2)
        downsample_factor = 4
        
        # Splitting data into training and validation sets
        x_train, x_valid, y_train, y_valid = self.split_data()

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


import tensorflow as tf
import keras
from keras import ops
from keras import layers

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = self.ctc_batch_cost

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

    def ctc_batch_cost(self, y_true, y_pred, input_length, label_length):
        label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
        input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
        sparse_labels = ops.cast(
            self.ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
        )

        y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

        return ops.expand_dims(
            tf.compat.v1.nn.ctc_loss(
                inputs=y_pred, labels=sparse_labels, sequence_length=input_length
            ),
            1,
        )

    def ctc_label_dense_to_sparse(self, labels, label_lengths):
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


CAPTCHA_TYPE = CaptchaType.NH_WEB_MAIL
WEIGHT_ONLY = False

print("A starting")
print("CAPTCHA_TYPE: ", CAPTCHA_TYPE)
print("WEIGHT_ONLY: ", WEIGHT_ONLY)
HYPER = Hyper(CaptchaType.NH_WEB_MAIL, WEIGHT_ONLY, quiet_out=False)
print("#### train info :", HYPER.image_width, HYPER.image_height, HYPER.max_length)
print("#### train characters :", HYPER.characters, len(HYPER.characters))

HYPER.train_model(epochs=100, earlystopping=True, early_stopping_patience=7)
