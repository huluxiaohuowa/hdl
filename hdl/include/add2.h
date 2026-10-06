// 作者：胡建星（Jianxing Hu）
// 邮箱：j.hu@pku.edu.cn
// 文件：hdl/include/add2.h
// 说明：自定义算子 C++/CUDA 头文件
// 模块功能：声明 add2 算子的核函数启动接口，供 C++ 绑定与 CUDA 实现两个翻译单元共享。
// 计算 c[i] = a[i] + b[i]，n 为元素个数；a、b 为只读输入，c 为就地写入的输出，三者均为 device 端 float 指针。
void launch_add2(float *c,
                 const float *a,
                 const float *b,
                 int n);