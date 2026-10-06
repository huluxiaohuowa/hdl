// 作者：胡建星（Jianxing Hu）
// 邮箱：j.hu@pku.edu.cn
// 文件：hdl/kernel/test.cu
// 说明：CUDA 自定义算子内核
// 模块功能：最小 CUDA 测试程序，仅启动一个空核函数以验证 nvcc 编译与设备端输出可用。
#include <stdio.h>

// 测试核函数：每个线程各打印一行文本，无输入输出参数
__global__ void hello (void)
{
    printf("kakakakakaka\n");
}

int main(void)
{
    // 核函数启动配置：1 个块、每块 10 个线程；cudaDeviceReset 释放设备上下文并刷新打印输出
    printf("hal");
    hello<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}