def zero_param_and_grad_with_string(model, txt):
    params = [
        param for param_name, param in model.named_parameters() if txt in param_name
    ]
    for param in params:
        param.data.zero_()
        param.requires_grad = False
