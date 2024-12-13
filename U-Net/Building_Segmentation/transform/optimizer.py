import torch.optim as optim

def get_optimizer(optimizer_name, parameters, lr=0.001, **kwargs):
  # kwargs: 옵티마이저에 적용할 인자들
  optimizers = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adagrad': optim.Adagrad,
    'AdamW': optim.AdamW
  }
  
  if optimizer_name not in optimizers:
    raise ValueError(f"Optmizer {optimizer_name}은 지원되지 않음")
  
  return optimizers[optimizer_name](parameters, lr=lr, **kwargs)