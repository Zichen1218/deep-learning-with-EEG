import torch
# ====== 1.device ======
device = (
    torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)

# ====== 2.hyperparameters ======
CONFIG = {
    'epochs':10,
    'learning_rate':1e-3,
    'hidden_dim':256,
    'batch_size':32,
    'dropout':0.3,
}