from hyper import CaptchaType, Hyper

CAPTCHA_TYPE = CaptchaType.NH_WEB_MAIL
PATIENCE = 7

Hyper().model_train(CAPTCHA_TYPE, PATIENCE)
