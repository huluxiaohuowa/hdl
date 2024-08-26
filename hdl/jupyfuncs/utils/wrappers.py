class GeneratorWrapper:
    def __init__(self, generator_func, *args, **kwargs):
        self.generator_func = generator_func
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.generator_func(*self.args, **self.kwargs)