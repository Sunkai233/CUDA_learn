# CUDA编程完整代码集 - 带文件名和编译命令

------

## 一、Hello World 程序

### 1. C++ Hello World

**文件名：** `hello.cpp`

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello World from C++!" << endl;
    return 0;
}
```

**编译命令：**

```bash
g++ hello.cpp -o hello
./hello
```

------

### 2. CUDA Hello World

**文件名：** `hello.cu`

```cpp
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

**编译命令：**

```bash
nvcc hello.cu -o hello
./hello
```

------

## 二、核函数基础示例

### 3. 多线程Hello World

**文件名：** `hello_threads.cu`

```cpp
#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Main function ending\n");
    return 0;
}
```

**编译命令：**

```bash
nvcc hello_threads.cu -o hello_threads
./hello_threads
```

------

### 4. 显示线程ID

**文件名：** `thread_id.cu`

```cpp
#include <stdio.h>

__global__ void test_printf() {
    printf("Thread %d: Hello\n", threadIdx.x);
}

int main() {
    test_printf<<<1, 5>>>();  // 5个线程
    cudaDeviceSynchronize();
    return 0;
}
```

**编译命令：**

```bash
nvcc thread_id.cu -o thread_id
./thread_id
```

------

### 5. 显示Block和Thread信息

**文件名：** `block_thread_id.cu`

```cpp
#include <stdio.h>

__global__ void my_kernel() {
    printf("Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    my_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

**编译命令：**

```bash
nvcc block_thread_id.cu -o block_thread_id
./block_thread_id
```

------

## 三、数组操作

### 6. C++串行数组加法

**文件名：** `array_add_cpu.cpp`

```cpp
#include <stdio.h>

void add_arrays(int* a, int* b, int* c, int n) {
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 10;
    int a[10], b[10], c[10];
    
    // 初始化
    for(int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    
    // 计算
    add_arrays(a, b, c, n);
    
    // 显示结果
    for(int i = 0; i < n; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }
    
    return 0;
}
```

**编译命令：**

```bash
g++ array_add_cpu.cpp -o array_add_cpu
./array_add_cpu
```

------

### 7. CUDA简单数组加法

**文件名：** `array_add_simple.cu`

```cpp
#include <stdio.h>

__global__ void add_arrays_gpu(int* a, int* b, int* c, int n) {
    int i = threadIdx.x;
    if(i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 10;
    int size = n * sizeof(int);
    
    // 主机内存
    int h_a[10], h_b[10], h_c[10];
    for(int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 设备内存
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 拷贝到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 启动核函数
    add_arrays_gpu<<<1, n>>>(d_a, d_b, d_c, n);
    
    // 拷贝回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 显示结果
    for(int i = 0; i < n; i++) {
        printf("c[%d] = %d\n", i, h_c[i]);
    }
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
```

**编译命令：**

```bash
nvcc array_add_simple.cu -o array_add_simple
./array_add_simple
```

------

### 8. 向量加法（大规模）

**文件名：** `vector_add.cu`

```cpp
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    int size = n * sizeof(float);
    
    // 分配主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 初始化
    for(int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2.0f;
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 拷贝到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 配置并启动
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    printf("Grid size: %d, Block size: %d\n", grid_size, block_size);
    
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    
    // 拷贝回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 验证前10个结果
    printf("Verification (first 10 elements):\n");
    for(int i = 0; i < 10; i++) {
        printf("c[%d] = %.0f (expected %.0f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

**编译命令：**

```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

------

### 9. 数组求和（原子操作）

**文件名：** `array_sum.cu`

```cpp
#include <stdio.h>

__global__ void array_sum_kernel(int* arr, int n, int* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < n) {
        atomicAdd(result, arr[idx]);
    }
}

int main() {
    int n = 1000;
    int size = n * sizeof(int);
    
    // 主机内存
    int* h_arr = new int[n];
    for(int i = 0; i < n; i++) h_arr[i] = 1;
    
    // 设备内存
    int *d_arr, *d_result;
    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_result, sizeof(int));
    
    // 拷贝数据
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));
    
    // 启动核函数
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    array_sum_kernel<<<grid_size, block_size>>>(d_arr, n, d_result);
    
    // 拷贝结果
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sum = %d (expected %d)\n", h_result, n);
    
    // 清理
    cudaFree(d_arr);
    cudaFree(d_result);
    delete[] h_arr;
    
    return 0;
}
```

**编译命令：**

```bash
nvcc array_sum.cu -o array_sum
./array_sum
```

------

## 四、内存管理示例

### 10. 内存管理演示

**文件名：** `memory_demo.cu`

```cpp
#include <stdio.h>

__global__ void simple_kernel(int* data) {
    int idx = threadIdx.x;
    data[idx] = data[idx] * 2;
}

int main() {
    int n = 10;
    int size = n * sizeof(int);
    
    // 1. CPU上准备数据
    int host_data[10];
    for(int i = 0; i < n; i++) {
        host_data[i] = i;
    }
    
    // 2. GPU上分配内存
    int* device_data;
    cudaMalloc(&device_data, size);
    
    // 3. CPU拷贝到GPU
    cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
    
    // 4. GPU执行
    simple_kernel<<<1, n>>>(device_data);
    
    // 5. GPU拷贝回CPU
    cudaMemcpy(host_data, device_data, size, cudaMemcpyDeviceToHost);
    
    // 6. 显示结果
    for(int i = 0; i < n; i++) {
        printf("data[%d] = %d\n", i, host_data[i]);
    }
    
    // 7. 清理
    cudaFree(device_data);
    
    return 0;
}
```

**编译命令：**

```bash
nvcc memory_demo.cu -o memory_demo
./memory_demo
```

------

## 五、高级特性示例

### 11. 共享内存计数器

**文件名：** `shared_memory_counter.cu`

```cpp
#include <stdio.h>

__global__ void counter() {
    __shared__ int shared_count;
    
    if(threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();
    
    atomicAdd(&shared_count, 1);
    __syncthreads();
    
    if(threadIdx.x == 0) {
        printf("Total count: %d\n", shared_count);
    }
}

int main() {
    counter<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

**编译命令：**

```bash
nvcc shared_memory_counter.cu -o shared_memory_counter
./shared_memory_counter
```

------

### 12. 模板核函数

**文件名：** `template_kernel.cu`

```cpp
#include <stdio.h>

template<int OPERATION>
__global__ void compute(int* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        if(OPERATION == 0) {
            data[idx] *= 2;
        } else if(OPERATION == 1) {
            data[idx] += 10;
        }
    }
}

int main() {
    int n = 10;
    int size = n * sizeof(int);
    
    int h_data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int *d_data;
    cudaMalloc(&d_data, size);
    
    // 测试乘法操作
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    compute<0><<<1, n>>>(d_data, n);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    printf("After multiplication by 2:\n");
    for(int i = 0; i < n; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");
    
    // 测试加法操作
    int h_data2[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    cudaMemcpy(d_data, h_data2, size, cudaMemcpyHostToDevice);
    compute<1><<<1, n>>>(d_data, n);
    cudaMemcpy(h_data2, d_data, size, cudaMemcpyDeviceToHost);
    
    printf("After adding 10:\n");
    for(int i = 0; i < n; i++) {
        printf("%d ", h_data2[i]);
    }
    printf("\n");
    
    cudaFree(d_data);
    return 0;
}
```

**编译命令：**

```bash
nvcc template_kernel.cu -o template_kernel
./template_kernel
```

------

### 13. 结构体参数传递

**文件名：** `struct_params.cu`

```cpp
#include <stdio.h>

struct KernelParams {
    int* data;
    int size;
    float factor;
};

__global__ void myKernel(KernelParams params) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < params.size) {
        params.data[idx] = (int)(params.data[idx] * params.factor);
    }
}

int main() {
    int n = 10;
    int size = n * sizeof(int);
    
    int h_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int *d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    KernelParams params;
    params.data = d_data;
    params.size = n;
    params.factor = 2.5f;
    
    myKernel<<<1, n>>>(params);
    
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    printf("Results:\n");
    for(int i = 0; i < n; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");
    
    cudaFree(d_data);
    return 0;
}
```

**编译命令：**

```bash
nvcc struct_params.cu -o struct_params
./struct_params
```

------

### 14. 异步执行演示

**文件名：** `async_demo.cu`

```cpp
#include <stdio.h>

__global__ void slow_kernel() {
    // 模拟耗时操作
    for(long long i = 0; i < 1000000000LL; i++);
    printf("GPU: Kernel完成\n");
}

int main() {
    printf("CPU: 准备启动kernel\n");
    
    slow_kernel<<<1, 1>>>();
    
    printf("CPU: Kernel已启动\n");
    
    printf("CPU: 做一些其他工作...\n");
    for(int i = 0; i < 100000000; i++);
    printf("CPU: 其他工作完成\n");
    
    cudaDeviceSynchronize();
    
    printf("CPU: 一切完成\n");
    return 0;
}
```

**编译命令：**

```bash
nvcc async_demo.cu -o async_demo
./async_demo
```

------

### 15. CPU-GPU协同工作

**文件名：** `cpu_gpu_overlap.cu`

```cpp
#include <stdio.h>

__global__ void gpu_compute(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        for(int i = 0; i < 1000; i++) {
            data[idx] = data[idx] * 1.01f;
        }
    }
}

void prepare_next_data(float* data, int n) {
    printf("CPU: 准备下一批数据...\n");
    for(int i = 0; i < n; i++) {
        data[i] = i * 2.0f;
    }
    printf("CPU: 数据准备完成\n");
}

int main() {
    int n = 1000000;
    int size = n * sizeof(float);
    
    float *h_data1 = (float*)malloc(size);
    float *h_data2 = (float*)malloc(size);
    float *d_data;
    
    for(int i = 0; i < n; i++) h_data1[i] = i;
    
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data1, size, cudaMemcpyHostToDevice);
    
    printf("启动GPU计算...\n");
    gpu_compute<<<(n+255)/256, 256>>>(d_data, n);
    
    // CPU在GPU计算的同时准备数据
    prepare_next_data(h_data2, n);
    
    printf("等待GPU完成...\n");
    cudaDeviceSynchronize();
    printf("GPU计算完成！\n");
    
    cudaFree(d_data);
    free(h_data1);
    free(h_data2);
    
    return 0;
}
```

**编译命令：**

```bash
nvcc cpu_gpu_overlap.cu -o cpu_gpu_overlap
./cpu_gpu_overlap
```

------

## 六、错误处理

### 16. CUDA错误检查

**文件名：** `error_check.cu`

```cpp
#include <stdio.h>

#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void simple_kernel(int* data) {
    int idx = threadIdx.x;
    data[idx] = idx;
}

int main() {
    int n = 10;
    int size = n * sizeof(int);
    int *d_data;
    
    // 使用错误检查宏
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));
    
    simple_kernel<<<1, n>>>(d_data);
    
    // 检查核函数启动错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 检查核函数执行错误
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    int h_data[10];
    CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < n; i++) {
        printf("data[%d] = %d\n", i, h_data[i]);
    }
    
    CHECK_CUDA_ERROR(cudaFree(d_data));
    
    printf("程序成功完成！\n");
    return 0;
}
```

**编译命令：**

```bash
nvcc error_check.cu -o error_check
./error_check
```

------

## 七、二维线程模型

### 17. 图像灰度化

**文件名：** `rgb_to_gray.cu`

```cpp
#include <stdio.h>
#include <stdlib.h>

__global__ void rgb_to_gray(unsigned char* rgb, unsigned char* gray, 
                           int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < width && y < height) {
        int idx = y * width + x;
        
        unsigned char r = rgb[idx * 3 + 0];
        unsigned char g = rgb[idx * 3 + 1];
        unsigned char b = rgb[idx * 3 + 2];
        
        gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main() {
    int width = 640, height = 480;
    int rgb_size = width * height * 3 * sizeof(unsigned char);
    int gray_size = width * height * sizeof(unsigned char);
    
    // 分配主机内存
    unsigned char *h_rgb = (unsigned char*)malloc(rgb_size);
    unsigned char *h_gray = (unsigned char*)malloc(gray_size);
    
    // 初始化RGB数据（模拟图像）
    for(int i = 0; i < width * height; i++) {
        h_rgb[i*3 + 0] = (i % 256);      // R
        h_rgb[i*3 + 1] = ((i*2) % 256);  // G
        h_rgb[i*3 + 2] = ((i*3) % 256);  // B
    }
    
    // 分配设备内存
    unsigned char *d_rgb, *d_gray;
    cudaMalloc(&d_rgb, rgb_size);
    cudaMalloc(&d_gray, gray_size);
    
    // 拷贝到设备
    cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice);
    
    // 配置线程
    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    
    printf("处理 %dx%d 图像\n", width, height);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", 
           grid_size.x, grid_size.y, block_size.x, block_size.y);
    
    // 启动核函数
    rgb_to_gray<<<grid_size, block_size>>>(d_rgb, d_gray, width, height);
    
    // 拷贝回主机
    cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);
    
    // 显示部分结果
    printf("\n前10个像素的灰度值:\n");
    for(int i = 0; i < 10; i++) {
        printf("gray[%d] = %d\n", i, h_gray[i]);
    }
    
    // 清理
    cudaFree(d_rgb);
    cudaFree(d_gray);
    free(h_rgb);
    free(h_gray);
    
    return 0;
}
```

**编译命令：**

```bash
nvcc rgb_to_gray.cu -o rgb_to_gray
./rgb_to_gray
```

------

### 18. 矩阵加法（完整版）

**文件名：** `matrix_add.cu`

```cpp
#include <stdio.h>
#include <stdlib.h>

__global__ void matrix_add(float* A, float* B, float* C, 
                          int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(col < width && row < height) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int width = 1024, height = 1024;
    int size = width * height * sizeof(float);
    
    // 1. 分配主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // 2. 初始化数据
    for(int i = 0; i < width * height; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    // 3. 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 4. 拷贝数据到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 5. 配置线程
    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    
    printf("矩阵大小: %d x %d\n", width, height);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", 
           grid_size.x, grid_size.y, block_size.x, block_size.y);
    printf("总线程数: %d\n", 
           grid_size.x * grid_size.y * block_size.x * block_size.y);
    
    // 6. 启动核函数
    matrix_add<<<grid_size, block_size>>>(d_A, d_B, d_C, width, height);
    
    // 7. 拷贝结果回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 8. 验证结果
    printf("\n验证前10个元素:\n");
    bool correct = true;
    for(int i = 0; i < 10; i++) {
        float expected = h_A[i] + h_B[i];
        printf("C[%d] = %.0f (expected %.0f) %s\n", 
               i, h_C[i], expected, (h_C[i] == expected) ? "✓" : "✗");
        if(h_C[i] != expected) correct = false;
    }
    printf("\n结果: %s\n", correct ? "全部正确！" : "有错误！");
    
    // 9. 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

**编译命令：**

```bash
nvcc matrix_add.cu -o matrix_add
./matrix_add
```

------

### 19. 矩阵乘法

**文件名：** `matrix_mul.cu`

```cpp
#include <stdio.h>
#include <stdlib.h>

__global__ void matrix_mul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N) {
        float sum = 0.0f;
        for(int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 512;  // 矩阵大小 N x N
    int size = N * N * sizeof(float);
    
    // 分配主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // 初始化
    for(int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 拷贝到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 配置线程
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    
    printf("计算 %dx%d 矩阵乘法\n", N, N);
    
    // 启动核函数
    matrix_mul<<<grid, block>>>(d_A, d_B, d_C, N);
    
    // 拷贝回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 验证（每个元素应该是 N * 1.0 * 2.0 = 2N）
    printf("C[0] = %.0f (expected %.0f)\n", h_C[0], (float)(N * 2));
    printf("C[1] = %.0f (expected %.0f)\n", h_C[1], (float)(N * 2));
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

**编译命令：**

```bash
nvcc matrix_mul.cu -o matrix_mul
./matrix_mul
```

------

## 八、性能测试

### 20. Block大小性能测试

**文件名：** `benchmark_block_size.cu`

```cpp
#include <stdio.h>
#include <chrono>

__global__ void compute_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        float x = data[idx];
        for(int i = 0; i < 100; i++) {
            x = x * 1.01f + 0.5f;
        }
        data[idx] = x;
    }
}

int main() {
    int n = 10000000;
    int size = n * sizeof(float);
    
    float *h_data = (float*)malloc(size);
    for(int i = 0; i < n; i++) h_data[i] = i;
    
    float *d_data;
    cudaMalloc(&d_data, size);
    
    int sizes[] = {32, 64, 128, 256, 512, 1024};
    
    printf("测试不同Block大小的性能:\n");
    printf("数据大小: %d elements\n\n", n);
    
    for(int i = 0; i < 6; i++) {
        int block_size = sizes[i];
        int grid_size = (n + block_size - 1) / block_size;
        
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        compute_kernel<<<grid_size, block_size>>>(d_data, n);
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        printf("Block size %4d: %8ld μs (Grid: %d blocks)\n", 
               block_size, duration.count(), grid_size);
    }
    
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

**编译命令：**

```bash
nvcc benchmark_block_size.cu -o benchmark_block_size
./benchmark_block_size
```

------

## 九、三维线程示例

### 21. 三维数据处理

**文件名：** `3d_array.cu`

```cpp
#include <stdio.h>

__global__ void process_3d(float* data, int Nx, int Ny, int Nz) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if(x < Nx && y < Ny && z < Nz) {
        int idx = z * Nx * Ny + y * Nx + x;
        data[idx] = x + y + z;
    }
}

int main() {
    int Nx = 64, Ny = 64, Nz = 64;
    int size = Nx * Ny * Nz * sizeof(float);
    
    float *h_data = (float*)malloc(size);
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // 配置三维线程
    dim3 block(8, 8, 8);  // 8x8x8 = 512个线程
    dim3 grid((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 7) / 8);
    
    printf("处理 %dx%dx%d 三维数组\n", Nx, Ny, Nz);
    printf("Block: (%d, %d, %d)\n", block.x, block.y, block.z);
    printf("Grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    
    process_3d<<<grid, block>>>(d_data, Nx, Ny, Nz);
    
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    // 验证几个点
    printf("\n验证结果:\n");
    printf("data[0,0,0] = %.0f (expected 0)\n", h_data[0]);
    printf("data[1,2,3] = %.0f (expected 6)\n", h_data[3*Nx*Ny + 2*Nx + 1]);
    
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

**编译命令：**

```bash
nvcc 3d_array.cu -o 3d_array
./3d_array
```

------

## 编译所有程序的脚本

### 22. 批量编译脚本

**文件名：** `compile_all.sh`

```bash
#!/bin/bash

echo "开始编译所有CUDA程序..."

# C++程序
g++ hello.cpp -o hello
g++ array_add_cpu.cpp -o array_add_cpu

# CUDA程序
nvcc hello.cu -o hello_cuda
nvcc hello_threads.cu -o hello_threads
nvcc thread_id.cu -o thread_id
nvcc block_thread_id.cu -o block_thread_id
nvcc array_add_simple.cu -o array_add_simple
nvcc vector_add.cu -o vector_add
nvcc array_sum.cu -o array_sum
nvcc memory_demo.cu -o memory_demo
nvcc shared_memory_counter.cu -o shared_memory_counter
nvcc template_kernel.cu -o template_kernel
nvcc struct_params.cu -o struct_params
nvcc async_demo.cu -o async_demo
nvcc cpu_gpu_overlap.cu -o cpu_gpu_overlap
nvcc error_check.cu -o error_check
nvcc rgb_to_gray.cu -o rgb_to_gray
nvcc matrix_add.cu -o matrix_add
nvcc matrix_mul.cu -o matrix_mul
nvcc benchmark_block_size.cu -o benchmark_block_size
nvcc 3d_array.cu -o 3d_array

echo "编译完成！"
echo "生成的可执行文件："
ls -lh hello hello_cuda hello_threads thread_id block_thread_id array_add_simple vector_add array_sum memory_demo shared_memory_counter template_kernel struct_params async_demo cpu_gpu_overlap error_check rgb_to_gray matrix_add matrix_mul benchmark_block_size 3d_array
```

**使用方法：**

```bash
chmod +x compile_all.sh
./compile_all.sh
```

------

## 清理脚本

### 23. 清理可执行文件

**文件名：** `clean.sh`

```bash
#!/bin/bash

echo "清理所有可执行文件..."

rm -f hello hello_cuda hello_threads thread_id block_thread_id
rm -f array_add_cpu array_add_simple vector_add array_sum
rm -f memory_demo shared_memory_counter template_kernel struct_params
rm -f async_demo cpu_gpu_overlap error_check
rm -f rgb_to_gray matrix_add matrix_mul
rm -f benchmark_block_size 3d_array

echo "清理完成！"
```

**使用方法：**

```bash
chmod +x clean.sh
./clean.sh
```

------

以上就是课程中所有代码的完整版本，每个文件都有明确的文件名和编译命令！