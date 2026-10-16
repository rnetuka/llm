class KVCache:

    def __init__(self, n_layers: int):
        self.cache = [None] * n_layers

    def get(self, i: int):
        return self.cache[i]

    def update(self, i: int, value):
        self.cache[i] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
