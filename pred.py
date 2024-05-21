from hyper import CaptchaType, Hyper

CAPTCHA_TYPE = CaptchaType.NH_WEB_MAIL
WEIGHT_ONLY = True

model = Hyper().model_validate(CAPTCHA_TYPE, WEIGHT_ONLY)
