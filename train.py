from PIL import Image
import core as cc

PATIENCE = 6
CAPTCHA_TYPE = cc.CaptchaType.NH_WEB_MAIL

train_img_path_list = cc.get_image_files(CAPTCHA_TYPE, train=True)
img = Image.open(train_img_path_list[0])
img_width = img.width
img_height = img.height

CM = cc.CreateModel(train_img_path_list, img_width, img_height)
model = CM.train_model(epochs=100, earlystopping=True, early_stopping_patience=PATIENCE)
weights_path = cc.get_weights_path(CAPTCHA_TYPE)
model.save_weights(weights_path)
