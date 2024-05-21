import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
from hyper import CaptchaType, Hyper, ApplyModel

START_TIME = time.time()
ARGV = sys.argv
CAPTCHA_TYPE = CaptchaType.SUPREME_COURT
WEIGHT_ONLY = True
HYPER = Hyper()
PRED = ""
END_TIME = None

def main(captchaType:CaptchaType, weight_only, imagePath:str):

    pred = ""

    try:
        pred = HYPER.predict(captchaType, weight_only, imagePath)
    except Exception as e:
        HYPER.quiet(False)
        print("Error:", e)

    return pred

if len(ARGV) < 3:
    print("Usage: " + os.path.basename(ARGV[0]) + " supreme_court|gov24|nh_web_mail IMAGE_FILE")
    sys.exit(-1)

if("__main__" == __name__):
    HYPER.quiet(True)
    CAPTCHA_TYPE = CaptchaType(ARGV[1])
    imagePath = ARGV[2]
    PRED = main(CAPTCHA_TYPE, WEIGHT_ONLY, imagePath)
    HYPER.quiet(False)
    print(PRED)
    END_TIME = time.time()
    print("time : ", END_TIME - START_TIME, "sec")
    print(PRED, CAPTCHA_TYPE, WEIGHT_ONLY, imagePath)
    sys.exit(0)

else:
    print("module imported")
