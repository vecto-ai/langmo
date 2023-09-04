def zero_and_freeze_param_by_name(model, name):
    params = [param for param_name, param in model.named_parameters() if name in param_name]
    for param in params:
        param.data.zero_()
        param.requires_grad = False
