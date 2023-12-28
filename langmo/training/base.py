from protonn.pl.cluster_mpi import MPIClusterEnvironment
from protonn.utils import get_time_str
from transformers import AutoTokenizer
from transformers import logging as tr_logging

from langmo.base import PLBase
from langmo.config.finetune import ConfigFinetune

# from langmo.nn.heads import get_downstream_head
from langmo.nn.utils import reinit_model, reinit_tensor


class BaseClassificationModel(PLBase):
    def forward(self, inputs):
        return self.net(**inputs)["logits"]

    # TODO: this seems to be wrong and also not used
    # def _compute_metric(self, logits, targets):
    #     return accuracy(F.softmax(logits, dim=1), targets)

    # def _compute_loss(self, logits, targets):
    #     return F.cross_entropy(logits, targets)


# class ClassificationFinetuner(BaseFinetuner):
#     def create_net(self):
#         return create_net(self.params)
