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
        """按 __init__ 里固化的 args/kwargs 现场调用被包装的生成器函数 generator_func，直接返回它给出的生成器（未预展开、不缓存产出），
        因此每次 for/in 本对象都会重新起一个全新的迭代序列。"""
        return self.generator_func(*self.args, **self.kwargs)