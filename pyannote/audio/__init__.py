class Pipeline:
    def __init__(self, *args, **kwargs):
        pass
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()
    def to(self, device):
        return self
