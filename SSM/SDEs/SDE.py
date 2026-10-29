class SDE:
    def __init__(self, dimension_state, dimension_noise):
        self.dimension_state = dimension_state
        self.dimension_noise = dimension_noise

    def drift(self, t, x):
        raise NotImplementedError

    def diffusion(self, t, x):
        raise NotImplementedError
