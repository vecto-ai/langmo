from langmo.config.base import CALLBACK_DEFAULTS, LangmoConfig


class ConfigPretrain(LangmoConfig):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["proba_masking"] = 0.12
        self.defaults["proba_random"] = 0.015
        self.defaults["proba_shortening"] = 0.1
        self.defaults["mask_special_tokens"] = True
        # TODO: we are not doing validation - so should remove this
        self.defaults["path_val_corpus"] = None
        # TODO: check that user's start from 0
        # TODO: check that the step is less then the key
        self.required_options.add("path_corpus")
        self.required_options.add("cnt_samples_per_epoch")
        self.defaults["callbacks"] = CALLBACK_DEFAULTS["mlm"]

    def _validate(self):
        super()._validate()
        if self["snapshot_schedule"] is not None:
            if list(self["snapshot_schedule"].keys())[0] != 0:
                raise Exception(f"snapshot_schedule should start from 0.")

    def get_run_name(self):
        name_run = self["model_name"]
        # TODO: revisit this when we have model parallel training
        name_run += f"_{self['timestamp']}"
        # name_run += f"_bs{params['batch_size'] * params['cnt_workers']}"
        name_run += f"_lr{self['max_lr']}"
        # TODO: add batch size
        # name_run += f"_wd{params['weight_decay']}"
        # name_run += f"_bs{params['batch_size_effective']}"
        return name_run
