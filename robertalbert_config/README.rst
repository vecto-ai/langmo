=========================
RoBERTAlbert Config Notes
=========================

This documents describes the creation of the configuration files to train 
"A Lite Roberta" that is a model with 
`Albert <https://arxiv.org/pdf/1909.11942>`_ architecture and 
`RoBERTa <https://arxiv.org/abs/1907.11692>`_ everything else **(mostly tokenizer)**.


RoBERTAlbert configs are created this way:
------------------------------------------

* All parameters only available for AlbertConfig are chosen from **albert-base-v2** config:
    ``AutoConfig.from_pretrained("albert-base-v2")``
* All parameters also available for RobertaConfig are then set to the values they have in **roberta-base** config:
    ``AutoConfig.from_pretrained("roberta-base")``

The following parameters are exceptions since they are kept as they are in **albert-base-v2** despite appearing in **roberta-base** as well:

* ``"_name_or_path"``: **"albert-base-v2"**
* ``"model_type"``: **"albert"**
* ``"architectures"``: **["AlbertForMaskedLM"]**
* ``"layer_norm_eps"``: **1e-12**


RoBERTa Tokenizer Note:
-----------------------

Internally Roberta adds 2 (this value depends on ``config.pad_token_id`` and can be avoided
if ``position_ids`` are provided explicitly to model forward) to all ``positional_token_ids`` so that the first 
two positional embeddings are kept free (to potentially add tokens), this motivates the 
``max_position_embeddings=514`` in ``AutoConfig.from_pretrained("roberta-base")``
(although the tokenizer defaults to max 512 length) the way the config is 
created adds this to RoBERTAlbert config which does not add this shift so the 
unused positional embeddings will be the last two or none.
`This is the line where this happens <https://github.com/huggingface/transformers/blob/048dd73bbacc30e62e7d895241c79b67db0b5751/src/transformers/models/roberta/modeling_roberta.py#L151>`_.


How To use:
-----------

Iside the folder with this readme lets call it ``this_folder`` there are the 
files to call:
::

    AutoModel.from_pretrained(this_folder)
    
    AutoTokenizer.from_pretrained(this_folder)
    # roberta-tokenizer saved in this folder 