// 作者：胡建星（Jianxing Hu）
// 邮箱：j.hu@pku.edu.cn
// 文件：hdl/kernel/add2_kernel.cu
// 说明：CUDA 自定义算子内核
// 模块功能：逐元素相加的 CUDA 核函数（CUDA kernel）add2_kernel 及其核函数启动配置（kernel launch configuration）。
__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    // 全局线程下标 i = 块索引 blockIdx.x * 每块线程数 blockDim.x + 线程索引 threadIdx.x；
    // 以整个网格的线程数 gridDim.x * blockDim.x 为步长循环，线程数少于 n 时也能覆盖全部元素，i >= n 时不访问显存
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

// 主机端启动函数：向上取整算出块个数（每块 1024 线程），把三个 device 指针与元素数传入核函数
void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    add2_kernel<<<grid, block>>>(c, a, b, n);
}