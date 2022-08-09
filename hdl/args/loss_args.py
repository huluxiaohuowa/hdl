from tap import Tap


class LossArgs(Tap):
    reduction: str = 'mean'