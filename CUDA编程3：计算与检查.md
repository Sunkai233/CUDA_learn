# CUDA并行编程课程详解

我会为你详细通俗地讲解这三个CUDA课程主题，并提供代码示例和类比。

------

## 一、组织线程模型（Thread Organization Model）

### 1. 核心概念

在CUDA中，我们需要组织成千上万个线程来并行处理数据。就像**组织一个大型工厂的工人**一样：

- **线程（Thread）** = 工人
- **线程块（Block）** = 车间
- **网格（Grid）** = 整个工厂

### 2. 数据存储方式

**重要概念**：数据在内存中是**线性存储**，按**行优先**方式排列。

假设有一个 16×8 的二维数组（16列，8行），在内存中实际是存储在连续的128个位置：

```
逻辑视图（二维）：
[0   1   2  ... 15 ]  <- 第0行
[16  17  18 ... 31 ]  <- 第1行
[32  33  34 ... 47 ]  <- 第2行
...

物理存储（一维）：
[0, 1, 2, 3, ..., 15, 16, 17, 18, ..., 127]
```

**类比**：就像读书，虽然书页是二维的（有行和列），但我们是按照从左到右、从上到下的顺序（一维）来阅读的。

------

### 3. 三种线程组织模型

#### 模型1：二维网格 + 二维线程块

**适用场景**：处理图像、矩阵等二维数据

**特点**：每个线程负责一个矩阵元素

```cpp
// 核函数示例
__global__ void processMatrix2D(float *data, int nx, int ny) {
    // 计算全局线程索引
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  // x方向的全局索引
    int iy = threadIdx.y + blockIdx.y * blockDim.y;  // y方向的全局索引
    
    // 转换为一维数组索引
    int idx = iy * nx + ix;
    
    // 边界检查
    if (ix < nx && iy < ny) {
        data[idx] = data[idx] * 2.0f;  // 示例操作
    }
}

// 主机代码调用
int main() {
    int nx = 16, ny = 8;
    
    // 定义线程块大小（每个块4×4=16个线程）
    dim3 block(4, 4);
    
    // 计算需要多少个块（向上取整）
    dim3 grid((nx + block.x - 1) / block.x,  // x方向需要4个块
              (ny + block.y - 1) / block.y); // y方向需要2个块
    
    processMatrix2D<<<grid, block>>>(d_data, nx, ny);
}
```

**可视化理解**：

```
Grid (4×2个块):
[Block(0,0)] [Block(1,0)] [Block(2,0)] [Block(3,0)]
[Block(0,1)] [Block(1,1)] [Block(2,1)] [Block(3,1)]

每个Block内部(4×4个线程):
[T(0,0)] [T(1,0)] [T(2,0)] [T(3,0)]
[T(0,1)] [T(1,1)] [T(2,1)] [T(3,1)]
[T(0,2)] [T(1,2)] [T(2,2)] [T(3,2)]
[T(0,3)] [T(1,3)] [T(2,3)] [T(3,3)]
```

------

#### 模型2：二维网格 + 一维线程块

**特点**：线程块是一维的，但网格是二维的

```cpp
__global__ void processMatrix1D(float *data, int nx, int ny) {
    // x方向索引计算相同
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    
    // y方向直接使用块索引（因为线程块是一维的）
    int iy = blockIdx.y;
    
    int idx = iy * nx + ix;
    
    if (ix < nx && iy < ny) {
        data[idx] = data[idx] * 2.0f;
    }
}

// 主机代码
int main() {
    int nx = 16, ny = 8;
    
    dim3 block(4);      // 一维线程块，4个线程
    dim3 grid((nx + block.x - 1) / block.x, ny);  // grid是二维的
    
    processMatrix1D<<<grid, block>>>(d_data, nx, ny);
}
```

------

#### 模型3：一维网格 + 一维线程块

**特点**：每个线程处理一整列数据，需要使用循环

```cpp
__global__ void processColumn(float *data, int nx, int ny) {
    // 只计算列索引
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (ix < nx) {
        // 循环处理该列的所有行
        for (int iy = 0; iy < ny; iy++) {
            int idx = iy * nx + ix;
            data[idx] = data[idx] * 2.0f;
        }
    }
}

// 主机代码
int main() {
    int nx = 16, ny = 8;
    
    int block = 4;                          // 一维块
    int grid = (nx + block - 1) / block;    // 一维网格
    
    processColumn<<<grid, block>>>(d_data, nx, ny);
}
```

**类比**：

- 模型1：每个工人负责一个零件
- 模型2：每个车间是一排工人，但车间排列是二维的
- 模型3：每个工人负责一整列零件（需要多次操作）

------

### 4. 索引计算总结

所有模型都遵循这个关键公式：

```cpp
// 全局索引 = 块内索引 + 块索引 × 块大小
ix = threadIdx.x + blockIdx.x * blockDim.x;
iy = threadIdx.y + blockIdx.y * blockDim.y;

// 转换为一维数组索引（行优先）
idx = iy * nx + ix;  // nx是每行的元素个数
```

------

## 二、运行时GPU信息查询

### 1. 查询GPU属性

CUDA提供了运行时API来查询GPU的各种信息。

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// 错误检查函数
void ErrorCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s at %s:%d\n", 
               cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CHECK(call) ErrorCheck(call, __FILE__, __LINE__)

int main() {
    int device_id = 0;  // GPU设备号
    cudaDeviceProp prop;
    
    // 查询GPU属性
    CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    // 打印GPU信息
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("Constant memory: %zu KB\n", prop.totalConstMem / 1024);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max grid size: (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max block size: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    
    return 0;
}
```

### 2. 查询CUDA核心数量

CUDA运行时API **不直接**提供查询核心数的函数，需要根据**计算能力（Compute Capability）**来计算。

```cpp
// 根据计算能力获取每个SM的CUDA核心数
int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1 || devProp.minor == 2) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            break;
        case 7: // Volta and Turing
            if (devProp.minor == 0 || devProp.minor == 5) cores = mp * 64;
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            break;
        default:
            printf("Unknown compute capability %d.%d!\n", 
                   devProp.major, devProp.minor);
            break;
    }
    
    return cores;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    
    int cores = getSPcores(prop);
    printf("Total CUDA cores: %d\n", cores);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("CUDA cores per SM: %d\n", cores / prop.multiProcessorCount);
    
    return 0;
}
```

------

## 三、CUDA计时

### 1. 使用CUDA事件计时

CUDA事件（Event）是最准确的GPU计时方法。

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) ErrorCheck(call, __FILE__, __LINE__)

// 简单的核函数示例
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20;  // 1M元素
    size_t bytes = n * sizeof(float);
    
    // 分配内存
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // ==================== CUDA事件计时 ====================
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    // 记录开始事件
    CHECK(cudaEventRecord(start));
    
    // ===== 需要计时的代码段 =====
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    // ===========================
    
    // 记录结束事件
    CHECK(cudaEventRecord(stop));
    
    // 等待事件完成
    CHECK(cudaEventSynchronize(stop));
    
    // 计算时间差
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    
    printf("执行时间: %.3f ms\n", elapsed_time);
    printf("带宽: %.2f GB/s\n", bytes * 3 / elapsed_time / 1e6);
    
    // 销毁事件
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    // 清理
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

**关键点**：

1. `cudaEventCreate()` - 创建事件
2. `cudaEventRecord()` - 记录事件时间点
3. `cudaEventSynchronize()` - 等待事件完成
4. `cudaEventElapsedTime()` - 计算两个事件之间的时间差（毫秒）
5. `cudaEventDestroy()` - 销毁事件

------

### 2. 使用nvprof性能分析

**nvprof**是NVIDIA提供的命令行性能分析工具（新版本推荐使用nsight systems）。

```bash
# 基本用法
nvprof ./your_program

# 详细分析
nvprof --print-gpu-trace ./your_program

# 分析特定指标
nvprof --metrics achieved_occupancy ./your_program

# 输出到文件
nvprof -o output.nvprof ./your_program
```

**输出示例**：

```
Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:   
  66.41%  38.016us    11   3.4560us  2.7520us  4.2560us  addFromGPU(...)
  17.80%  10.240us     3   3.4130us  3.3920us  3.4240us  [CUDA memcpy HtoD]
  10.73%   6.1440us    3   2.0480us  1.6320us  2.6880us  [CUDA memset]

API calls:
  95.18%  1.10411s     3  368.04ms  2.2000us  1.1041s   cudaMalloc
   3.79%  43.908ms     1  43.908ms  43.908ms  43.908ms  cudaDeviceReset
```

------

### 3. 完整示例：对比不同线程组织的性能

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

#define N 4096
#define CHECK(call) ErrorCheck(call, __FILE__, __LINE__)

void ErrorCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("Error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

// 2D Grid, 2D Block
__global__ void kernel2D2D(float *data) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * N + ix;
    
    if (ix < N && iy < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

// 2D Grid, 1D Block
__global__ void kernel2D1D(float *data) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = blockIdx.y;
    int idx = iy * N + ix;
    
    if (ix < N && iy < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

// 1D Grid, 1D Block
__global__ void kernel1D1D(float *data) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (ix < N) {
        for (int iy = 0; iy < N; iy++) {
            int idx = iy * N + ix;
            data[idx] = data[idx] * 2.0f;
        }
    }
}

float testKernel(void (*kernel)(float*), dim3 grid, dim3 block, 
                 float *d_data, const char *name) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    CHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(d_data);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float time;
    CHECK(cudaEventElapsedTime(&time, start, stop));
    
    printf("%s: %.3f ms\n", name, time);
    
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    return time;
}

int main() {
    size_t bytes = N * N * sizeof(float);
    float *d_data;
    CHECK(cudaMalloc(&d_data, bytes));
    
    // 测试1: 2D Grid + 2D Block
    dim3 block1(16, 16);
    dim3 grid1((N + block1.x - 1) / block1.x, (N + block1.y - 1) / block1.y);
    testKernel(kernel2D2D, grid1, block1, d_data, "2D Grid + 2D Block");
    
    // 测试2: 2D Grid + 1D Block
    dim3 block2(256);
    dim3 grid2((N + block2.x - 1) / block2.x, N);
    testKernel(kernel2D1D, grid2, block2, d_data, "2D Grid + 1D Block");
    
    // 测试3: 1D Grid + 1D Block
    dim3 block3(256);
    dim3 grid3((N + block3.x - 1) / block3.x);
    testKernel(kernel1D1D, grid3, block3, d_data, "1D Grid + 1D Block");
    
    cudaFree(d_data);
    return 0;
}
```

------

## 总结

### 线程组织模型选择建议：

| 应用场景 | 推荐模型           | 原因                       |
| -------- | ------------------ | -------------------------- |
| 图像处理 | 2D Grid + 2D Block | 直观，每个线程对应一个像素 |
| 矩阵运算 | 2D Grid + 2D Block | 符合数学表示               |
| 向量运算 | 1D Grid + 1D Block | 简单高效                   |
| 列处理   | 1D Grid + 1D Block | 每个线程处理一列           |

### 性能优化要点：

1. **合理设置Block大小**：通常使用128、256或512
2. **确保线程数是32的倍数**（Warp大小）
3. **使用CUDA事件精确计时**
4. **用nvprof分析瓶颈**
5. **检查内存访问模式**（合并访问）

希望这个详细的讲解对你有帮助！如有疑问欢迎继续提问。