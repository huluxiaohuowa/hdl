#include <stdio.h>

__global__ void hello (void)
{
    printf("kakakakakaka\n");
}

int main(void)
{
    printf("hal");
    hello<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}