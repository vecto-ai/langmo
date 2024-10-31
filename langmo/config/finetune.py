from langmo.config.base import LangmoConfig


class ConfigFinetune(LangmoConfig):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["siamese"] = False
        self.defaults["freeze_encoder"] = False
        self.defaults["encoder_wrapper"] = "pooler"
        self.defaults["shuffle"] = False
        self.defaults["cnt_seps"] = -1
        self.defaults["save_predictions"] = False
