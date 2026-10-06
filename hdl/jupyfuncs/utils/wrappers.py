# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/utils/wrappers.py
# 说明：通用包装与装饰工具
class GeneratorWrapper:
    def __init__(self, generator_func, *args, **kwargs):
        self.generator_func = generator_func
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.generator_func(*self.args, **self.kwargs)