// 作者：胡建星（Jianxing Hu）
// 邮箱：j.hu@pku.edu.cn
// 文件：hdl/pytorch/add2_ops.cpp
// 说明：自定义 add2 算子的 PyTorch C++ 扩展绑定（前向/反向与自动微分注册）
// 模块功能：CUDA 算子 add2 的 PyTorch 扩展绑定，把 device 指针封装成可被 Python 调用的函数。
#include <torch/extension.h>
#include "add2.h"

// 前向函数：从 torch::Tensor 取出 data_ptr 并转成 float*，转发给 launch_add2 就地写 c；
// n 为元素个数，本算子只注册前向，未实现反向传播（backward）。
void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_add2((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

// pybind11 模块入口：以名字 torch_launch_add2 把上面的 C++ 函数暴露给 Python 扩展模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}

// 注册到 dispatcher：使算子可通过 torch.ops.add2.torch_launch_add2 调用
TORCH_LIBRARY(add2, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}