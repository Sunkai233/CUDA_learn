# CUDA内存模型与GPU硬件资源详解

------

## 一、GPU硬件资源

### 1. 流多处理器（SM - Streaming Multiprocessor）

**核心概念**：SM是GPU实现并行计算的基础硬件单元。

**类比**：如果把GPU比作一个大工厂，那么：

- **SM** = 车间（每个车间独立运作）
- **CUDA Core** = 工人（实际干活的）
- **Warp Scheduler** = 车间主管（分配任务）

#### Fermi架构SM的关键资源

```
一个SM包含：
├── 32个CUDA核心（计算单元）
├── 寄存器文件（Register File）- 32K个32位寄存器
├── 共享内存/L1缓存 - 64KB（可配置）
├── 加载/存储单元（LD/ST Units）
├── 特殊函数单元（SFU）- 处理sin、cos等
└── Warp调度器 - 管理线程束执行
```

#### SM的工作方式

```cpp
// 示例：理解SM如何执行线程块

__global__ void simpleKernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    // 假设GPU有4个SM
    int n = 1024;
    dim3 block(128);  // 每个块128个线程
    dim3 grid(8);     // 8个块
    
    // 执行时：8个块会被分配到4个SM上
    // 每个SM可能同时处理2个块
    simpleKernel<<<grid, block>>>(d_data, n);
    
    return 0;
}
```

**重要特性**：

1. **每个SM支持数百个线程并发执行**
2. **线程块一旦分配到某个SM，就不会迁移到其他SM**
3. **一个SM可以同时执行多个线程块**（资源允许的情况下）

------

### 2. 线程模型与物理结构的映射

#### 软件视角 vs 硬件视角

| 维度     | 软件模型（逻辑）   | 硬件结构（物理）  |
| -------- | ------------------ | ----------------- |
| 最小单位 | Thread（线程）     | CUDA Core         |
| 组织单位 | Block（线程块）    | SM（流多处理器）  |
| 执行单位 | **Warp（线程束）** | **32个CUDA Core** |
| 整体     | Grid（网格）       | Device（整个GPU） |

```cpp
// 理解软件到硬件的映射

__global__ void demonstrateMapping(int *result) {
    // 软件层面
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int globalId = threadId + blockId * blockDim.x;
    
    // 硬件层面（自动完成）
    // 1. 这个线程属于哪个warp？
    int warpId = threadId / 32;
    
    // 2. 在warp中的位置
    int laneId = threadId % 32;
    
    result[globalId] = warpId * 1000 + laneId;
}

int main() {
    // 启动配置
    dim3 block(128);  // 128个线程 = 4个warp
    dim3 grid(2);     // 2个块
    
    // 实际执行：
    // - 2个块可能分配到不同的SM
    // - 每个块的128个线程分成4个warp
    // - 每个warp（32线程）在硬件上真正并行执行
    
    demonstrateMapping<<<grid, block>>>(d_result);
    
    return 0;
}
```

**关键理解**：

```
软件定义的线程块 → 硬件上分配到SM → 自动分割成多个Warp

例如：block(256) 被分配到SM后
↓
自动分割成 8 个 Warp
Warp 0: 线程 0-31
Warp 1: 线程 32-63
Warp 2: 线程 64-95
...
Warp 7: 线程 224-255
```

------

### 3. 线程束（Warp）- GPU并行的真正秘密

#### 什么是Warp？

**定义**：32个连续的线程组成一个warp，是GPU硬件调度和执行的基本单位。

**SIMT架构**（Single Instruction, Multiple Threads）：

- 一个warp中的32个线程**执行相同的指令**
- 但可以处理**不同的数据**

```cpp
// Warp的形成规则示例

__global__ void showWarpFormation() {
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;  // lane = warp内的位置
    
    printf("Thread %d: Warp %d, Lane %d\n", tid, warpId, laneId);
}

int main() {
    // 测试1：128个线程
    showWarpFormation<<<1, 128>>>();
    // 结果：形成 4 个warp（128/32 = 4）
    
    // 测试2：100个线程
    showWarpFormation<<<1, 100>>>();
    // 结果：形成 4 个warp（向上取整：ceil(100/32) = 4）
    // 最后一个warp只有4个活跃线程，其余28个空闲！
    
    cudaDeviceSynchronize();
    return 0;
}
```

#### Warp计算公式

```cpp
// 计算线程块需要多少个warp
int numWarps = (blockDim.x + 31) / 32;  // 向上取整

// 或者使用标准库
#include <math.h>
int numWarps = (int)ceil((float)blockDim.x / 32.0f);
```

#### Warp效率问题

**重要**：线程块大小应该是32的倍数！

```cpp
// 好的配置
dim3 block1(128);   // 128/32 = 4个warp，100%利用率
dim3 block2(256);   // 256/32 = 8个warp，100%利用率
dim3 block3(64);    // 64/32 = 2个warp，100%利用率

// 不好的配置
dim3 block4(100);   // ceil(100/32) = 4个warp
                    // 最后一个warp只用了4/32 = 12.5%
                    // 浪费了28个线程的资源！

dim3 block5(50);    // ceil(50/32) = 2个warp
                    // 最后一个warp只用了18/32 = 56.25%
```

#### Warp分歧（Warp Divergence）

**问题**：当warp内的线程执行不同的代码路径时，会导致性能下降。

```cpp
// 示例：Warp分歧问题

__global__ void warpDivergenceExample(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // ❌ 坏例子：奇偶线程执行不同操作
        if (idx % 2 == 0) {
            // 偶数线程执行这里
            data[idx] = data[idx] * 2;
        } else {
            // 奇数线程执行这里
            data[idx] = data[idx] + 10;
        }
        // 问题：同一个warp内的线程走了不同分支
        // GPU必须先执行if分支，再执行else分支
        // 相当于串行执行，损失了50%性能！
    }
}

// ✅ 好例子：避免warp分歧
__global__ void noWarpDivergence(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // 方案1：使用无分支的算术运算
        int isOdd = idx & 1;
        data[idx] = data[idx] * (2 - isOdd * 2) + (isOdd * 10);
        
        // 或者方案2：确保同一warp的线程走相同分支
        // 因为warp是32个连续线程，可以按warp边界对齐分支
    }
}
```

**分歧检测代码**：

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkDivergence(int *flags) {
    int tid = threadIdx.x;
    int warpId = tid / 32;
    
    // 统计warp内有多少线程进入if分支
    int vote = 0;
    
    if (tid % 2 == 0) {
        vote = 1;
    }
    
    // 使用warp内的同步原语统计
    unsigned mask = 0xffffffff;
    int total = __popc(__ballot_sync(mask, vote));
    
    if (tid % 32 == 0) {  // 每个warp的第一个线程
        printf("Warp %d: %d/32 threads took 'if' branch\n", warpId, total);
        flags[warpId] = total;
    }
}

int main() {
    int *d_flags;
    cudaMalloc(&d_flags, 4 * sizeof(int));
    
    checkDivergence<<<1, 128>>>(d_flags);
    cudaDeviceSynchronize();
    
    cudaFree(d_flags);
    return 0;
}
```

------

## 二、CUDA内存模型概述

### 1. 内存层次结构特点

**局部性原理**：

- **时间局部性**：最近访问的数据很可能再次被访问
- **空间局部性**：访问某个数据后，其附近的数据也可能被访问

#### 内存金字塔

```
速度快 ↑              容量小 ↑
       |                     |
   寄存器 (Registers)         |
       ↓                     |
    L1缓存                    |
       ↓                     |
  共享内存 (Shared Memory)     |
       ↓                     |
    L2缓存                    |
       ↓                     |
  全局内存 (Global Memory)     |
       ↓                     |
    主机内存                  ↓
       |                     |
速度慢 ↓              容量大 ↓
```

**硬件实现**：

- SRAM

  （静态随机存储器）：寄存器、缓存、共享内存

  - 速度快，成本高，容量小

- DRAM

  （动态随机存储器）：全局内存、主机内存

  - 速度慢，成本低，容量大

------

### 2. CUDA六大内存类型

| 内存类型     | 物理位置       | 访问权限 | 可见范围      | 生命周期       | 速度           |
| ------------ | -------------- | -------- | ------------- | -------------- | -------------- |
| **寄存器**   | 片上(On-chip)  | 读写     | 单个线程      | 线程生命周期   | ⚡最快          |
| **本地内存** | 片外(Off-chip) | 读写     | 单个线程      | 线程生命周期   | 🐌慢            |
| **共享内存** | 片上(On-chip)  | 读写     | 单个线程块    | 线程块生命周期 | ⚡很快          |
| **全局内存** | 片外(Off-chip) | 读写     | 所有线程+主机 | 主机分配/释放  | 🐌很慢          |
| **常量内存** | 片外(Off-chip) | 只读     | 所有线程+主机 | 主机分配/释放  | 中等（有缓存） |
| **纹理内存** | 片外(Off-chip) | 只读     | 所有线程+主机 | 主机分配/释放  | 中等（有缓存） |

------

### 3. 寄存器（Register）

#### 特点

```cpp
__global__ void registerExample() {
    // ✅ 这些变量存储在寄存器中
    int a = 10;
    float b = 3.14f;
    double c = 2.718;  // 需要2个寄存器（64位）
    
    // 内建变量也在寄存器中
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 计算
    float result = a * b + c;
}
```

#### 寄存器限制

```cpp
// 查询寄存器使用情况
// 编译时添加：nvcc --ptxas-options=-v kernel.cu

// 输出示例：
// ptxas info : Used 24 registers, 384 bytes cmem[0]
```

**不同架构的寄存器数量**：

| 计算能力           | 每个SM的寄存器数 | 每个线程最大寄存器数 |
| ------------------ | ---------------- | -------------------- |
| 2.x (Fermi)        | 32K              | 63                   |
| 3.x (Kepler)       | 64K              | 255                  |
| 5.x (Maxwell)      | 64K              | 255                  |
| 6.x (Pascal)       | 64K              | 255                  |
| 7.x (Volta/Turing) | 64K              | 255                  |
| 8.x (Ampere)       | 64K              | 255                  |

```cpp
// 限制每个线程使用的寄存器数量
__global__ void __launch_bounds__(256, 4)  // maxThreadsPerBlock, minBlocksPerSM
myKernel() {
    // ...
}

// 或者在编译时指定
// nvcc -maxrregcount=32 kernel.cu
```

------

### 4. 本地内存（Local Memory）

#### 什么时候使用本地内存？

```cpp
__global__ void localMemoryExample() {
    // ❌ 这些会放到本地内存（不是寄存器）
    
    // 1. 编译时无法确定索引的数组
    int arr[100];
    int idx = threadIdx.x % 7;  // 动态索引
    arr[idx] = 42;
    
    // 2. 占用过多寄存器的大型结构体
    struct LargeStruct {
        float data[200];
    } myStruct;
    
    // 3. 寄存器溢出的变量
    float var1, var2, var3; // ...定义了300个变量
}

// ✅ 这些会放到寄存器
__global__ void registerStorageExample() {
    // 1. 编译时可确定索引的数组
    int arr[4];
    arr[0] = 1;
    arr[1] = 2;
    arr[2] = 3;
    arr[3] = 4;
    
    // 2. 简单变量
    int x = threadIdx.x;
    float y = x * 2.0f;
}
```

**重要**：本地内存虽然名字叫"本地"，但实际上是全局内存的一部分，速度很慢！

```cpp
// 检查本地内存使用
// nvcc --ptxas-options=-v kernel.cu

// 输出示例：
// ptxas info : Used 24 registers, 128 bytes lmem, 384 bytes cmem[0]
//                                  ↑
//                            本地内存使用量
```

------

### 5. 寄存器溢出（Register Spilling）

#### 为什么会溢出？

```cpp
// 示例：寄存器溢出场景

__global__ void spillExample() {
    // 假设这个核函数需要80个寄存器/线程
    
    float a[20];
    float b[20];
    float c[20];
    
    // 大量计算...
    for (int i = 0; i < 20; i++) {
        c[i] = a[i] * b[i] + a[i] - b[i];
    }
}

int main() {
    // 如果启动配置是这样：
    dim3 block(256);  // 256个线程/块
    
    // 每个SM有64K寄存器
    // 如果一个SM同时运行2个块 = 512个线程
    // 需要的寄存器 = 512 * 80 = 40,960个
    // 这在限制内，不会溢出
    
    // 但如果启动4个块 = 1024个线程
    // 需要的寄存器 = 1024 * 80 = 81,920个
    // 超过了64K = 65,536个
    // 导致寄存器溢出到本地内存！
    
    spillExample<<<grid, block>>>();
    
    return 0;
}
```

#### 避免寄存器溢出的方法

```cpp
// 方法1：减少每个线程的寄存器使用
__global__ void optimizedKernel() {
    // 重用变量，避免定义过多临时变量
    float temp;
    
    // 而不是
    // float temp1, temp2, temp3, ...
}

// 方法2：使用编译选项限制
// nvcc -maxrregcount=32 kernel.cu

// 方法3：使用launch bounds
__global__ void __launch_bounds__(128)  // 每个块最多128线程
myKernel() {
    // 编译器会根据这个信息优化寄存器分配
}

// 方法4：减小线程块大小
dim3 block(128);  // 而不是256或512
```

#### 监控寄存器溢出

```cpp
// 使用nvprof检查
// nvprof --metrics local_memory_overhead ./my_program

// 或者在代码中查询
cudaFuncAttributes attr;
cudaFuncGetAttributes(&attr, myKernel);
printf("Local memory per thread: %zu bytes\n", attr.localSizeBytes);
printf("Registers per thread: %d\n", attr.numRegs);
```

------

## 三、完整示例：内存使用对比

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// 纯寄存器版本
__global__ void registerOnly(float *output) {
    int tid = threadIdx.x;
    
    // 所有变量都在寄存器中
    float a = tid * 2.0f;
    float b = tid + 1.0f;
    float c = a * b;
    
    output[tid] = c;
}

// 可能使用本地内存的版本
__global__ void withLocalMemory(float *output) {
    int tid = threadIdx.x;
    
    // 动态索引的数组 → 本地内存
    float arr[10];
    int idx = tid % 10;
    
    for (int i = 0; i < 10; i++) {
        arr[i] = i * tid;
    }
    
    output[tid] = arr[idx];
}

// 使用共享内存的版本（下节课详解）
__global__ void withSharedMemory(float *output) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    shared[tid] = tid * 2.0f;
    
    __syncthreads();
    
    output[tid] = shared[tid];
}

int main() {
    float *d_output;
    cudaMalloc(&d_output, 256 * sizeof(float));
    
    // 测试不同版本
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 版本1：寄存器
    cudaEventRecord(start);
    registerOnly<<<1, 256>>>(d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time1;
    cudaEventElapsedTime(&time1, start, stop);
    printf("Register only: %.3f ms\n", time1);
    
    // 版本2：本地内存
    cudaEventRecord(start);
    withLocalMemory<<<1, 256>>>(d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time2;
    cudaEventElapsedTime(&time2, start, stop);
    printf("With local memory: %.3f ms\n", time2);
    
    // 版本3：共享内存
    cudaEventRecord(start);
    withSharedMemory<<<1, 256>>>(d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time3;
    cudaEventElapsedTime(&time3, start, stop);
    printf("With shared memory: %.3f ms\n", time3);
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_output);
    
    return 0;
}
```

------

## 总结与最佳实践

### 关键要点

1. **SM是GPU的核心**
   - 理解SM的资源限制
   - 合理分配线程块
2. **Warp是执行单位**
   - 线程块大小应为32的倍数
   - 避免warp分歧
3. **内存层次很重要**
   - 优先使用寄存器
   - 避免本地内存
   - 后续学习共享内存优化
4. **监控资源使用**
   - 使用nvcc的详细输出
   - 用nvprof/nsight分析性能

### 优化建议

```cpp
// ✅ 好的实践
dim3 block(256);        // 32的倍数
dim3 grid((n + 255) / 256);

// ❌ 避免
dim3 block(100);        // 浪费warp资源
dim3 block(1024);       // 超过硬件限制
```

希望这份详细讲解帮助你理解CUDA的内存模型和GPU硬件！有任何问题欢迎继续提问。