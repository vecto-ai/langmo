from langmo.config.base import LangmoConfig


class ConfigPretrain(LangmoConfig):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["proba_masking"] = 0.12
        self.defaults["proba_random"] = 0.015
        self.defaults["mask_special_tokens"] = True
        self.required_options.add("path_corpus")
        self.required_options.add("path_val_corpus")
        self.required_options.add("cnt_samples_per_epoch")
