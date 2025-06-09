import types, sys
class _Cuda:
    def is_available(self):
        return False
    def empty_cache(self):
        pass

cuda = _Cuda()

functional = types.ModuleType('torch.nn.functional')

nn = types.ModuleType('torch.nn')
nn.functional = functional

sys.modules['torch.nn'] = nn
sys.modules['torch.nn.functional'] = functional

def manual_seed(seed):
    pass
class device:
    def __init__(self, name):
        self.name = name
