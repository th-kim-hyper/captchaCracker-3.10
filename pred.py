from hyper import CaptchaType, Hyper

CAPTCHA_TYPE = CaptchaType.NH_WEB_MAIL
WEIGHT_ONLY = False

HYPER = Hyper(CaptchaType.GOV24)
HYPER.model_validate(CAPTCHA_TYPE, WEIGHT_ONLY)
# model = Hyper().model_validate(CAPTCHA_TYPE, WEIGHT_ONLY)
