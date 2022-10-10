from .networks.mil_net import MILNet
from .networks.multimil_net import MultiMILNet

model_dict = {
    'mil': {'name': MILNet,
            'params': ['depth', 'frozen_stages', 'net_name', 'pool_mode', 'topk']
            },
    'multimil': {'name': MultiMILNet,
            'params': ['depth', 'frozen_stages', 'net_name', 'pool_mode', 'topk']
            }
}
