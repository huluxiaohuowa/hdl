# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/dl/list.py
# 说明：深度学习张量与模型辅助工具
# 模块功能：通用列表集合运算工具（交集/并集/差集），与化学或神经网络逻辑无关。
def list_diff(listA, listB, mode="intersection"):
    """Calculate the difference between two lists based on the specified mode.
    
        Args:
            listA (list): The first list.
            listB (list): The second list.
            mode (str, optional): The mode to determine the difference. 
                Possible values are "intersection" (default), "union", or "diff".
    
        Returns:
            list: A list containing the elements based on the specified mode.
    """
    if mode == "intersection":
        ret = list(set(listA).intersection(set(listB)))
    elif mode == "union":
        ret = list(set(listA).union(set(listB)))
    elif mode == "diff":
        ret = list(set(listB).difference(set(listA)))
    
    return ret