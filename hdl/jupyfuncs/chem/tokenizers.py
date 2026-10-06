# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/chem/tokenizers.py
# 说明：化学信息学工具（RDKit / Jupyter）
# 模块功能：SMILES 词元化（tokenization）用的正则表达式，供模型输入按 token 切分 SMILES 字符串。
# 正则逐个 token 匹配：方括号原子 [...]、Br/Cl、N O S P F I、小写芳香原子 b c n o s p、
# 括号与点、键符 = # / \、电荷 + -、环闭合 %nn 与单个数字，以及其他符号（@ : ? > * $ 等）；
# 用 re.findall 匹配即可得到 SMILES 字符串的 token 序列。
SMI_REGEX_PATTERN = \
    "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"
