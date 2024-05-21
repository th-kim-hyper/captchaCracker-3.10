import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import core as cc
from PIL import Image
import glob
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CAPTCHA_TYPE = cc.CaptchaType.SUPREME_COURT
ARGV = sys.argv
NULL_OUT = open(os.devnull, 'w')
ORI_OUT = sys.stdout
BASE_DIR = cc.get_base_dir()

def main(captchaType:cc.CaptchaType, imagePath:str):

    pred = ""
    baseDir = BASE_DIR

    try:
        img = Image.open(os.path.join(baseDir, imagePath))
        img_width = img.width
        img_height = img.height

        weights_path = os.path.join(baseDir, "model", captchaType.value + ".weights.h5")
        train_img_dir = os.path.join(baseDir, "images", captchaType.value, "train")
        train_img_path_list = glob.glob(train_img_dir + os.sep + "*.png")
        labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in train_img_path_list]
        max_length = max([len(label) for label in labels])
        characters = sorted(set(char for label in labels for char in label))

        AM = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)
        pred = AM.predict(imagePath)
    except Exception as e:
        sys.stdout = ORI_OUT
        print("Error:", e)

    return pred

if len(ARGV) < 3:
    print("Usage: " + os.path.basename(ARGV[0]) + " supreme_court|gov24|nh_web_mail IMAGE_FILE")
    sys.exit(-1)

if("__main__" == __name__):
    sys.stdout = NULL_OUT
    CAPTCHA_TYPE = cc.CaptchaType(ARGV[1])
    imagePath = ARGV[2]
    pred = main(CAPTCHA_TYPE, imagePath)
    sys.stdout = ORI_OUT
    print(pred)
    sys.exit(0)

else:
    print("module imported")
