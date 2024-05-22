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

CAPTCHA_TYPE = CaptchaType.GOV24
WEIGHT_ONLY = False

print("A starting")
print("CAPTCHA_TYPE: ", CAPTCHA_TYPE)
print("WEIGHT_ONLY: ", WEIGHT_ONLY)
HYPER = Hyper(CaptchaType.NH_WEB_MAIL, quiet_out=False)
print("#### train info :", HYPER.image_width, HYPER.image_height, HYPER.max_length)
print("#### train characters :", HYPER.characters, len(HYPER.characters))

HYPER
