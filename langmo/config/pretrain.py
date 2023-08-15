from langmo.config.base import LangmoConfig, CALLBACK_DEFAULTS


class ConfigPretrain(LangmoConfig):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["proba_masking"] = 0.12
        self.defaults["proba_random"] = 0.015
        self.defaults["proba_shortening"] = 0.1
        self.defaults["mask_special_tokens"] = True
        self.defaults["path_val_corpus"] = None
        self.required_options.add("path_corpus")
        self.required_options.add("cnt_samples_per_epoch")
        self.defaults["callbacks"] = CALLBACK_DEFAULTS["mlm"]
