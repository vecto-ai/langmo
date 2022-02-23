import pytorch_lightning as pl


class LayerNormCallback(pl.Callback):
    def setup(self, *args, **kwargs):
        self.layernorm_storage = dict()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not batch_idx % (10 * pl_module.hparams["log_every_n_steps"]) == 0:
            return
        for layer in [1, 4, 8, 11]:
            if "roberta" in pl_module.hparams["model_name"]:
                layernorm_weight = pl_module.net.get_parameter(
                    f"roberta.encoder.layer.{layer}.output.LayerNorm.weight"
                )
                layernorm_bias = pl_module.net.get_parameter(
                    f"roberta.encoder.layer.{layer}.output.LayerNorm.bias"
                )
            elif "bert" in pl_module.hparams["model_name"]:
                layernorm_weight = pl_module.net.get_parameter(
                    f"bert.encoder.layer.{layer}.output.LayerNorm.weight"
                )
                layernorm_bias = pl_module.net.get_parameter(
                    f"bert.encoder.layer.{layer}.output.LayerNorm.bias"
                )

            layernorm_weight = (
                layernorm_weight.data.clone().detach().cpu()
            )
            layernorm_bias = layernorm_bias.data.clone().detach().cpu()

            pl_module.log_dict(
                {
                    f"weight_layer_{layer}/val_{i}": val.item()
                    for i, val in enumerate(layernorm_weight)
                },
                rank_zero_only=True,
            )
            pl_module.log_dict(
                {
                    f"bias_layer_{layer}/val_{i}": val.item()
                    for i, val in enumerate(layernorm_bias)
                },
                rank_zero_only=True,
            )
