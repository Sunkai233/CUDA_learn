# 从C++编程到CUDA编程 - 课程详解

这是一个关于CUDA并行编程的入门课件，主讲人是权双老师。让我为您详细讲解这个课程的核心内容。

------

## 📚 课程概述

这门课程旨在帮助学习者从传统的C++编程过渡到CUDA并行编程，是CUDA并行编程系列课程的一部分。

------

## 第一部分：C++中的Hello World

### C++程序开发的三个步骤

**1. 编写源代码**

- 使用文本编辑器编写源代码，推荐使用 VSCode
- 也可以使用其他任何文本编辑器，如 Vim 等
- 源代码文件通常以 `.cpp` 为扩展名

**2. 编译程序**

- 编译器对源码进行一系列操作：
  - **预处理**：处理 `#include`、`#define` 等预处理指令
  - **编译**：将源代码转换为机器代码
  - **链接**：将多个目标文件和库文件链接成可执行文件
- C++中使用 **G++** 编译器

**3. 运行可执行文件**

- 执行编译生成的二进制文件

------

## 第二部分：编译C++程序

### 安装G++编译器

在 Ubuntu 系统下，使用以下命令安装 G++：

bash

```bash
sudo apt-get install g++
```

### 编译命令

基本的编译命令格式：

bash

```bash
g++ hello.cpp -o hello
```

**命令解析：**

- `g++`：调用G++编译器
- `hello.cpp`：源代码文件名
- `-o hello`：指定输出的可执行文件名为 `hello`

------

## 第三部分：CUDA中的Hello World程序

### NVCC编译器介绍

**NVCC** 是 NVIDIA 的 CUDA 编译器，它具有以下特点：

**1. 安装方式**

- 安装 CUDA Toolkit 后即可使用 nvcc 编译器
- CUDA Toolkit 包含了完整的开发环境

**2. 兼容性**

- nvcc 完全支持纯C++代码的编译
- 这意味着您可以用 nvcc 编译普通的 `.cpp` 文件

**3. CUDA文件编译**

- 专门用于编译扩展名为 `.cu` 的 CUDA 源文件
- `.cu` 文件可以包含主机代码（CPU代码）和设备代码（GPU代码）

### 编译CUDA程序

编译CUDA文件的命令格式：

bash

```bash
nvcc hello.cu -o hello
```

**命令解析：**

- `nvcc`：调用 NVCC 编译器
- `hello.cu`：CUDA源代码文件（注意扩展名是 `.cu`）
- `-o hello`：指定输出的可执行文件名为 `hello`

------

## 🔑 关键要点总结

### C++ vs CUDA 编译对比

```
特性C++CUDA
文件扩展名.cpp.cu
编译器g++nvcc
编译命令g++ file.cpp -o outputnvcc file.cu -o output
运行平台CPUCPU + GPU
语法兼容性标准C++C++ + CUDA扩展
```

### 从C++到CUDA的过渡

1. **语法相似性**：CUDA是C++的扩展，基本语法保持一致
2. **编译工具**：从 g++ 过渡到 nvcc
3. **编程模型**：从串行编程过渡到并行编程
4. **执行模型**：从CPU执行扩展到CPU+GPU协同执行

------





# CUDA核函数 - 深度讲解

让我为您深入讲解CUDA核函数的概念、原理和实践应用。

------

## 🎯 为什么需要核函数？

### 从串行到并行的思维转变

在传统的C++编程中，我们习惯于这样思考：

cpp

```cpp
// C++串行思维：一步一步执行
void add_arrays(int* a, int* b, int* c, int n) {
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // 一次处理一个元素
    }
}
```

**问题**：当n = 1,000,000时，需要执行100万次循环，效率很低。

**CUDA的解决方案**：让100万个线程同时工作，每个线程处理一个元素！

cpp

```cpp
// CUDA并行思维：所有线程同时执行
__global__ void add_arrays_gpu(int* a, int* b, int* c, int n) {
    int i = threadIdx.x;  // 每个线程知道自己的编号
    if(i < n) {
        c[i] = a[i] + b[i];  // 100万个线程同时执行这一行
    }
}
```

这就是核函数的魔力：**把循环变成并行执行**！

------

## 第一部分：深入理解核函数

### 1.1 核函数的本质

核函数不是一个普通的函数，它是一个**模板**，告诉GPU：

1. **做什么**（函数体里的代码）
2. **谁来做**（由多少个线程执行）
3. **怎么做**（每个线程执行相同的代码，但处理不同的数据）

### 1.2 `__global__` 限定词的深层含义

cpp

```cpp
__global__ void myKernel() {
    // 这段代码会在GPU上运行
}
```

CUDA中有三个重要的限定词：

```
限定词在哪执行从哪调用实际用途
__global__GPUCPU核函数 - 并行计算的入口
__device__GPUGPU设备函数 - 被核函数调用的辅助函数
__host__CPUCPU主机函数 - 普通C++函数（默认）
```

**形象比喻**：

- `__global__`：总经理（从外部调用，统领全局）
- `__device__`：部门经理（内部协作，辅助工作）
- `__host__`：其他公司的人（完全独立的系统）

### 1.3 为什么返回值必须是void？

这是一个经常被问到的问题。原因有两个：

**原因1：异步执行**

cpp

```cpp
// CPU调用核函数
myKernel<<<1, 1>>>();  // CPU立即返回，不等GPU执行完
printf("kernel已启动\n");  // 这行可能比GPU先执行
```

如果核函数有返回值，CPU要等多久才能拿到？这会破坏异步执行的优势。

**原因2：多线程并行**

cpp

```cpp
// 假设有1000个线程执行这个核函数
myKernel<<<1, 1000>>>();
```

如果每个线程都返回一个值，应该返回哪个？所有1000个返回值怎么合并？这在逻辑上不成立。

**正确的做法**：通过指针参数传递结果

cpp

```cpp
__global__ void compute(int* input, int* output) {
    int idx = threadIdx.x;
    output[idx] = input[idx] * 2;  // 每个线程写入自己的结果
}
```

------

## 第二部分：核函数的5大限制详解

### 2.1 只能访问GPU内存 - 内存隔离的原理

这是最重要也最容易出错的限制。

**错误示例**：

cpp

~~~cpp
int main() {
    int host_data = 42;  // 这是CPU内存中的变量
    
    myKernel<<<1, 1>>>(&host_data);  // ❌ 严重错误！
    // GPU无法访问CPU内存，程序会崩溃
}
```

**为什么会这样？**

CPU和GPU有各自独立的内存系统：
```
CPU 内存 (RAM)          GPU 内存 (显存)
┌─────────────┐        ┌─────────────┐
│ host_data   │  ❌    │             │
│             │ 无法  │             │
│             │ 访问  │             │
└─────────────┘        └─────────────┘
~~~

**正确做法**：

cpp

```cpp
int main() {
    // 1. 在CPU上准备数据
    int host_data = 42;
    
    // 2. 在GPU上分配内存
    int* device_data;
    cudaMalloc(&device_data, sizeof(int));
    
    // 3. 将数据从CPU拷贝到GPU
    cudaMemcpy(device_data, &host_data, sizeof(int), 
               cudaMemcpyHostToDevice);
    
    // 4. GPU现在可以访问这个数据了
    myKernel<<<1, 1>>>(device_data);  // ✅ 正确
    
    // 5. 清理
    cudaFree(device_data);
}
```

**记忆口诀**：CPU的给CPU，GPU的给GPU，中间靠拷贝。

### 2.2 不能使用变长参数 - 编译器限制

**什么是变长参数？**

cpp

```cpp
// C语言中的变长参数函数
void my_printf(const char* fmt, ...) {
    // ... 可以接受任意数量的参数
}

my_printf("值: %d, %d, %d\n", 1, 2, 3);  // 3个参数
my_printf("值: %d\n", 1);                // 1个参数
```

**为什么核函数不支持？**

CUDA编译器需要在编译时就确定每个线程需要的寄存器和内存资源。变长参数会导致：

1. 无法预测内存使用量
2. 无法优化寄存器分配
3. 影响并行执行效率

**替代方案**：

cpp

```cpp
// 使用结构体传递多个参数
struct KernelParams {
    int* data;
    int size;
    float factor;
};

__global__ void myKernel(KernelParams params) {
    // 使用 params.data, params.size, params.factor
}
```

### 2.3 不能使用静态变量 - 生命周期问题

**错误示例**：

cpp

```cpp
__global__ void counter() {
    static int count = 0;  // ❌ 错误！
    count++;
    printf("count = %d\n", count);
}
```

**为什么不行？**

假设我们启动1000个线程：

cpp

```cpp
counter<<<1, 1000>>>();
```

**问题1**：1000个线程同时访问同一个静态变量，会产生竞争条件（race condition）

**问题2**：静态变量应该在哪个线程中初始化？

**问题3**：静态变量的生命周期如何管理？核函数执行完后是否保留？

**正确做法 - 使用共享内存**：

cpp

```cpp
__global__ void counter() {
    __shared__ int shared_count;  // 共享内存，线程块内共享
    
    if(threadIdx.x == 0) {
        shared_count = 0;  // 由第一个线程初始化
    }
    __syncthreads();  // 等待初始化完成
    
    // 现在可以安全使用了（需要原子操作避免竞争）
    atomicAdd(&shared_count, 1);
}
```

### 2.4 不能使用函数指针 - 灵活性vs性能

**C++中的函数指针**：

cpp

```cpp
void func1() { printf("Function 1\n"); }
void func2() { printf("Function 2\n"); }

void execute(void (*fp)()) {
    fp();  // 调用函数指针指向的函数
}
```

**为什么核函数不支持？**

1. **性能开销**：函数指针调用无法内联优化，会严重影响GPU性能
2. **控制流分歧**：不同线程调用不同函数会导致线程束分歧
3. **编译复杂性**：GPU的函数调用机制与CPU不同

**替代方案 - 使用模板**：

cpp

```cpp
template<int OPERATION>
__global__ void compute(int* data) {
    int idx = threadIdx.x;
    if(OPERATION == 0) {
        data[idx] *= 2;
    } else if(OPERATION == 1) {
        data[idx] += 10;
    }
}

// 调用
compute<0><<<1, 100>>>(data);  // 执行乘法操作
compute<1><<<1, 100>>>(data);  // 执行加法操作
```

### 2.5 异步性 - 最重要的特性

这是核函数最容易被误解，但也是最强大的特性。

**异步执行示例**：

cpp

~~~cpp
#include <stdio.h>

__global__ void slow_kernel() {
    // 模拟耗时操作
    for(int i = 0; i < 1000000000; i++);
    printf("GPU: Kernel完成\n");
}

int main() {
    printf("CPU: 准备启动kernel\n");
    
    slow_kernel<<<1, 1>>>();  // 启动GPU计算
    
    printf("CPU: Kernel已启动\n");  // 这行会立即执行！
    
    // CPU继续做其他事情
    printf("CPU: 做一些其他工作...\n");
    for(int i = 0; i < 100000000; i++);
    printf("CPU: 其他工作完成\n");
    
    cudaDeviceSynchronize();  // 等待GPU完成
    
    printf("CPU: 一切完成\n");
    return 0;
}
```

**输出**：
```
CPU: 准备启动kernel
CPU: Kernel已启动
CPU: 做一些其他工作...
CPU: 其他工作完成
GPU: Kernel完成
CPU: 一切完成
```

**时间线图示**：
```
CPU时间线: ──启动kernel──>继续执行──>等待同步──>
                ↓                      ↑
GPU时间线:      └──执行kernel(耗时)────┘
~~~

**实际应用 - CPU和GPU协同工作**：

cpp

```cpp
// 在GPU计算的同时，CPU可以做准备工作
gpu_compute<<<grid, block>>>(data1);  // 启动GPU计算
prepare_next_data(data2);              // CPU准备下一批数据
cudaDeviceSynchronize();               // 等GPU完成
```

------

## 第三部分：CUDA程序编写流程深度解析

### 3.1 完整的Hello World程序解析

让我们逐行分析这个程序：

cpp

```cpp
#include <stdio.h>

// 核函数定义 - 在GPU上执行的代码
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    // 第10行：调用核函数
    // <<<1, 1>>>: 1个线程块，每个块1个线程
    hello_from_gpu<<<1, 1>>>();
    
    // 第11行：同步等待
    cudaDeviceSynchronize();
    
    // 第12行：返回
    return 0;
}
```

### 3.2 执行配置 `<<<>>>` 详解

这是CUDA最独特的语法：

cpp

```cpp
kernel<<<grid_dim, block_dim, shared_mem, stream>>>(args);
          ↑         ↑           ↑          ↑
          |         |           |          |
       线程块数  每块线程数  共享内存   流(可选)
```

**简化版本（最常用）**：

cpp

```cpp
kernel<<<grid_dim, block_dim>>>(args);
```

**实例1：单线程执行**

cpp

```cpp
hello<<<1, 1>>>();  // 1个线程块，1个线程
// 相当于只有1个线程在工作
```

**实例2：并行执行**

cpp

```cpp
hello<<<1, 256>>>();  // 1个线程块，256个线程
// 256个线程同时执行hello函数
```

**实例3：多线程块**

cpp

```cpp
hello<<<10, 256>>>();  // 10个线程块，每块256个线程
// 总共2560个线程同时工作
```

### 3.3 为什么需要 `cudaDeviceSynchronize()`？

**实验：没有同步会发生什么？**

cpp

~~~cpp
#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_from_gpu<<<1, 1>>>();
    // 注意：没有cudaDeviceSynchronize()
    printf("Main function ending\n");
    return 0;  // 程序立即结束
}
```

**可能的输出**：
```
Main function ending
~~~

GPU的输出消失了！因为程序在GPU执行完之前就结束了。

**加上同步**：

cpp

~~~cpp
int main() {
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();  // 等待GPU完成
    printf("Main function ending\n");
    return 0;
}
```

**输出**：
```
Hello from GPU!
Main function ending
~~~

### 3.4 核函数中的printf特殊之处

虽然核函数不支持iostream，但支持printf，不过有一些特殊之处：

cpp

~~~cpp
__global__ void test_printf() {
    printf("Thread %d: Hello\n", threadIdx.x);
}

int main() {
    test_printf<<<1, 5>>>();  // 5个线程
    cudaDeviceSynchronize();
}
```

**可能的输出**（顺序不确定）：
```
Thread 2: Hello
Thread 0: Hello
Thread 4: Hello
Thread 1: Hello
Thread 3: Hello
~~~

**注意**：

1. 输出顺序是不确定的（并行执行）
2. printf有缓冲区大小限制
3. 大量printf会严重影响性能

------

## 🔬 实战演练：从C++到CUDA的转换

### 示例：数组求和

**C++版本（串行）**：

cpp

```cpp
void array_sum(int* arr, int n) {
    int sum = 0;
    for(int i = 0; i < n; i++) {
        sum += arr[i];  // 逐个累加
    }
    printf("Sum = %d\n", sum);
}
```

**CUDA版本（并行 - 初级）**：

cpp

```cpp
__global__ void array_sum_kernel(int* arr, int n, int* result) {
    int idx = threadIdx.x;
    
    if(idx < n) {
        // 每个线程将自己的元素加到结果中
        atomicAdd(result, arr[idx]);  // 原子操作避免冲突
    }
}

int main() {
    int n = 1000;
    int* h_arr = new int[n];  // CPU数组
    int* d_arr;               // GPU数组
    int* d_result;            // GPU结果
    
    // 初始化数组
    for(int i = 0; i < n; i++) h_arr[i] = 1;
    
    // 在GPU上分配内存
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));  // 结果初始化为0
    
    // 启动核函数：1000个线程并行工作
    array_sum_kernel<<<1, n>>>(d_arr, n, d_result);
    
    // 拷贝结果回CPU
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sum = %d\n", h_result);  // 输出：Sum = 1000
    
    // 清理
    cudaFree(d_arr);
    cudaFree(d_result);
    delete[] h_arr;
}
```

**关键对比**：

```
方面C++串行CUDA并行
执行方式1个线程循环1000次1000个线程同时执行
时间复杂度O(n)O(1) 理论上
内存管理简单需要显式管理
代码复杂度低较高
```

------

## 💡 常见陷阱和调试技巧

### 陷阱1：忘记同步

cpp

```cpp
__global__ void kernel() { /* ... */ }

int main() {
    kernel<<<1, 1>>>();
    // 忘记cudaDeviceSynchronize()
    return 0;  // 程序可能在GPU完成前就结束
}
```

**解决**：养成习惯，核函数调用后立即同步（除非你明确知道在做什么）。

### 陷阱2：传递CPU指针给GPU

cpp

```cpp
int* ptr = new int[100];  // CPU内存
kernel<<<1, 1>>>(ptr);    // ❌ 灾难性错误
```

**解决**：使用`cudaMalloc`分配GPU内存。

### 陷阱3：线程数超出限制

cpp

```cpp
// 每个线程块最多1024个线程（大多数GPU）
kernel<<<1, 2048>>>();  // ❌ 超出限制
```

**解决**：分成多个线程块

cpp

```cpp
kernel<<<2, 1024>>>();  // ✅ 正确：2个块，每块1024线程
```

### 调试技巧

**1. 使用cuda-memcheck检测内存错误**：

bash

```bash
cuda-memcheck ./my_program
```

**2. 检查CUDA错误**：

cpp

```cpp
cudaError_t err = cudaMalloc(&ptr, size);
if(err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

**3. 简化测试**：

cpp

~~~cpp
// 先用少量线程测试
kernel<<<1, 1>>>();  // 单线程，便于调试

// 确认无误后再扩展
kernel<<<grid, block>>>();  // 大规模并行
```

---

## 🎓 学习路线图
```
第1阶段：基础
├─ 理解核函数概念 ✓（本节课）
├─ 掌握基本语法
└─ 学会内存管理

第2阶段：线程组织
├─ Grid、Block、Thread层次
├─ 线程索引计算
└─ 多维线程组织

第3阶段：内存优化
├─ 全局内存、共享内存
├─ 内存合并访问
└─ Bank冲突避免

第4阶段：性能优化
├─ 占用率优化
├─ 指令级优化
└─ 算法并行化
~~~

------

通过本节课的深入讲解，您应该已经掌握了CUDA核函数的核心概念、使用限制和编程技巧。核函数是CUDA编程的基石，理解它的特性对后续学习至关重要。下一节课，我们将学习如何组织和管理成千上万个线程，真正释放GPU的并行计算能力！



# CUDA线程模型 - 深度讲解

这是CUDA并行编程系列课程的第三讲，将深入讲解CUDA的线程组织模型。这是CUDA编程中最重要也是最容易混淆的概念之一。

------

## 📋 课程目录

1. **线程模型结构** - 理解Grid和Block的层次结构
2. **线程组织管理** - 掌握线程索引和标识计算
3. **网格和线程块限制** - 了解硬件约束

------

## 🎯 为什么需要线程模型？

在深入之前，让我们先理解为什么CUDA需要这样的线程组织方式。

### 现实世界的类比

想象你是一个工厂的经理，需要组织10,000名工人完成任务：

**方案A：扁平化管理（没有层次）**

```
经理 → 10,000名工人
```

问题：无法有效管理，沟通困难，资源分配混乱

**方案B：层次化管理（CUDA的方式）**

```
经理 → 10个车间主任 → 每个车间1,000名工人
```

优势：便于管理、资源分配、任务协调

CUDA的线程模型就是采用这种层次化的组织方式！

------

## 第一部分：线程模型结构深度解析

### 1.1 核心概念：Grid（网格）和Block（线程块）

CUDA线程组织采用**二级层次结构**：

```
Grid (网格)
├─── Block 0 (线程块0)
│    ├─── Thread 0
│    ├─── Thread 1
│    └─── Thread N
├─── Block 1 (线程块1)
│    ├─── Thread 0
│    ├─── Thread 1
│    └─── Thread N
└─── Block M (线程块M)
     └─── ...
```

### 1.2 关键术语解释

**Grid（网格）**：

- 所有线程块的集合
- 一个核函数调用对应一个Grid
- 相当于"整个工厂"

**Block（线程块）**：

- 一组线程的集合
- 相当于"一个车间"
- 同一个Block内的线程可以协作（共享内存、同步）

**Thread（线程）**：

- 执行核函数的最小单元
- 相当于"一个工人"
- 每个线程执行相同的代码，但处理不同的数据

### 1.3 物理 vs 逻辑的重要区分

⚠️ **关键理解**：线程分块是**逻辑上**的划分，**物理上**线程不分块！

这是什么意思？让我详细解释：

**逻辑层面（程序员的视角）**：

```
Grid有很多Block，每个Block有很多Thread
Block之间是独立的，可以以任意顺序执行
```

**物理层面（GPU硬件的视角）**：

```
所有线程都在同一个GPU上
GPU以32个线程为单位（称为warp）调度执行
Block的划分只是为了编程方便和资源管理
```

**形象比喻**：

```
逻辑划分：就像公司的组织架构图（部门、小组）
物理执行：所有人实际上都在同一栋大楼里工作
```

### 1.4 配置线程：执行配置详解

**基本语法**：

cpp

```cpp
kernel<<<grid_size, block_size>>>(args);
```

**实例1：最简单的配置**

cpp

```cpp
// 1个线程块，1个线程
kernel<<<1, 1>>>();

// 相当于：
// Grid: 1个Block
// Block 0: 1个Thread
// 总共只有1个线程工作
```

**实例2：单Block多线程**

cpp

```cpp
// 1个线程块，256个线程
kernel<<<1, 256>>>();

// 相当于：
// Grid: 1个Block
// Block 0: 256个Thread
// 这256个线程可以协作（共享内存、同步）
```

**实例3：多Block配置**

cpp

```cpp
// 10个线程块，每块256个线程
kernel<<<10, 256>>>();

// 相当于：
// Grid: 10个Block
// 每个Block: 256个Thread
// 总共 10 × 256 = 2,560 个线程
```

### 1.5 硬件限制（重要！）

**线程块大小限制**：

- 最大值：**1024个线程**
- 这是单个Block能包含的最大线程数
- 超过这个值会导致编译或运行时错误

**网格大小限制（一维）**：

- 最大值：**2³¹ - 1 = 2,147,483,647** 个Block
- 这个限制非常大，实际应用中很少达到

**实际例子**：

cpp

```cpp
// ✅ 正确
kernel<<<1, 1024>>>();      // 刚好1024个线程
kernel<<<1000, 512>>>();    // 1000个块，每块512线程

// ❌ 错误
kernel<<<1, 2048>>>();      // 超过1024限制！
kernel<<<1, 1025>>>();      // 超过1024限制！
```

------

## 第二部分：一维线程模型详解

### 2.1 内建变量（Built-in Variables）

CUDA提供了一组特殊的变量，让每个线程知道自己的"身份"：

```
变量类型含义取值范围
gridDim.xint网格中Block的数量等于执行配置中的grid_size
blockDim.xintBlock中Thread的数量等于执行配置中的block_size
blockIdx.xint当前Block在Grid中的索引0 到 gridDim.x-1
threadIdx.xint当前Thread在Block中的索引0 到 blockDim.x-1
```

⚠️ **重要特性**：

- 这些变量**无需定义**，在核函数中直接可用
- 每个线程看到的值**不同**（threadIdx和blockIdx）
- 只在核函数中有效，主机代码中不可用

### 2.2 具体示例：kernel<<<2, 4>>>()

让我们详细分析这个配置：

cpp

~~~cpp
__global__ void my_kernel() {
    printf("Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    my_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
}
```

**配置分析**：
- `gridDim.x` = 2 （2个线程块）
- `blockDim.x` = 4 （每块4个线程）
- `blockIdx.x` 范围：0 ~ 1
- `threadIdx.x` 范围：0 ~ 3

**内存布局**：
```
Grid
├─── Block 0 (blockIdx.x = 0)
│    ├─── Thread 0 (threadIdx.x = 0)
│    ├─── Thread 1 (threadIdx.x = 1)
│    ├─── Thread 2 (threadIdx.x = 2)
│    └─── Thread 3 (threadIdx.x = 3)
└─── Block 1 (blockIdx.x = 1)
     ├─── Thread 0 (threadIdx.x = 0)  ← 注意：threadIdx又从0开始
     ├─── Thread 1 (threadIdx.x = 1)
     ├─── Thread 2 (threadIdx.x = 2)
     └─── Thread 3 (threadIdx.x = 3)
~~~

### 2.3 全局唯一线程标识（核心公式！）

**问题**：Block 0的Thread 1和Block 1的Thread 1都有threadIdx.x = 1，如何区分？

**解决方案**：计算全局唯一的线程ID

**一维线程全局ID公式**：

cpp

~~~cpp
int global_id = threadIdx.x + blockIdx.x * blockDim.x;
```

**公式推导**：
```
Block 0的线程全局ID：0, 1, 2, 3
Block 1的线程全局ID：4, 5, 6, 7
Block 2的线程全局ID：8, 9, 10, 11
...

计算方式：
Block 0, Thread 0: 0 + 0 × 4 = 0
Block 0, Thread 1: 1 + 0 × 4 = 1
Block 1, Thread 0: 0 + 1 × 4 = 4  ← 跳到下一个Block
Block 1, Thread 1: 1 + 1 × 4 = 5
~~~

**实战应用 - 数组加法**：

cpp

~~~cpp
__global__ void vector_add(float* a, float* b, float* c, int n) {
    // 计算全局线程ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 边界检查（防止越界）
    if(idx < n) {
        c[idx] = a[idx] + b[idx];  // 每个线程处理一个元素
    }
}

int main() {
    int n = 1000000;  // 100万个元素
    
    // 每个Block 256个线程
    int block_size = 256;
    
    // 需要多少个Block？向上取整
    int grid_size = (n + block_size - 1) / block_size;  // 3907个Block
    
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
}
```

**为什么这个公式有效？**

想象一维数组的分块：
```
数组: [0][1][2][3][4][5][6][7][8][9][10][11]
      └─ Block 0 ─┘└─ Block 1 ─┘└─ Block 2 ─┘
      blockSize=4   blockSize=4   blockSize=4

线程映射：
Block 0, Thread 0 → 数组[0]
Block 0, Thread 3 → 数组[3]
Block 1, Thread 0 → 数组[4]  ← 需要跳过前面的Block
Block 1, Thread 1 → 数组[5]
~~~

------

## 第三部分：多维线程模型

### 3.1 为什么需要多维？

一维线程模型适合处理一维数据（如向量），但很多问题是多维的：

**应用场景**：

- 图像处理：2D像素网格
- 矩阵运算：2D矩阵
- 3D体素处理：3D体积数据
- 物理模拟：3D空间

**使用多维的好处**：

1. 代码更直观（二维索引对应二维数据）
2. 内存访问模式更清晰
3. 避免手动计算多维到一维的转换

### 3.2 CUDA支持的维度

CUDA可以组织**三维**的网格和线程块：

cpp

```cpp
// dim3是一个结构体，有x, y, z三个成员
dim3 grid_size(Gx, Gy, Gz);    // 网格：Gx × Gy × Gz 个Block
dim3 block_size(Bx, By, Bz);   // 线程块：Bx × By × Bz 个Thread
```

### 3.3 多维内建变量

**索引变量（uint3类型）**：

cpp

```cpp
blockIdx.x, blockIdx.y, blockIdx.z   // Block在Grid中的3D索引
threadIdx.x, threadIdx.y, threadIdx.z // Thread在Block中的3D索引
```

**维度变量（dim3类型）**：

cpp

```cpp
gridDim.x, gridDim.y, gridDim.z      // Grid各维度的大小
blockDim.x, blockDim.y, blockDim.z   // Block各维度的大小
```

**取值范围**：

cpp

```cpp
blockIdx.x ∈ [0, gridDim.x - 1]
blockIdx.y ∈ [0, gridDim.y - 1]
blockIdx.z ∈ [0, gridDim.z - 1]

threadIdx.x ∈ [0, blockDim.x - 1]
threadIdx.y ∈ [0, blockDim.y - 1]
threadIdx.z ∈ [0, blockDim.z - 1]
```

### 3.4 默认值规则

**重要特性**：未指定的维度默认为1

cpp

```cpp
// 只指定x维度
kernel<<<10, 256>>>();

// 等价于：
dim3 grid(10, 1, 1);
dim3 block(256, 1, 1);
kernel<<<grid, block>>>();

// 内建变量的值：
// gridDim.x = 10, gridDim.y = 1, gridDim.z = 1
// blockDim.x = 256, blockDim.y = 1, blockDim.z = 1
```

### 3.5 二维线程模型示例

**示例：2×2网格，5×3线程块**

cpp

~~~cpp
dim3 grid_size(2, 2);      // 等价于 dim3(2, 2, 1)
dim3 block_size(5, 3);     // 等价于 dim3(5, 3, 1)

kernel<<<grid_size, block_size>>>();
```

**可视化理解**：
```
Grid: 2×2 = 4个Block
┌─────────────┬─────────────┐
│  Block(0,0) │  Block(1,0) │
│   5×3线程   │   5×3线程   │
├─────────────┼─────────────┤
│  Block(0,1) │  Block(1,1) │
│   5×3线程   │   5×3线程   │
└─────────────┴─────────────┘

每个Block内部: 5×3 = 15个Thread
┌───┬───┬───┬───┬───┐
│0,0│1,0│2,0│3,0│4,0│  ← threadIdx.y = 0
├───┼───┼───┼───┼───┤
│0,1│1,1│2,1│3,1│4,1│  ← threadIdx.y = 1
├───┼───┼───┼───┼───┤
│0,2│1,2│2,2│3,2│4,2│  ← threadIdx.y = 2
└───┴───┴───┴───┴───┘
  ↑
  threadIdx.x
~~~

### 3.6 二维线程索引计算

**Block内的线程索引（一维化）**：

cpp

~~~cpp
int tid = threadIdx.y * blockDim.x + threadIdx.x;
```

**推导过程**（以5×3的Block为例）：
```
Thread(0,0): 0 × 5 + 0 = 0
Thread(1,0): 0 × 5 + 1 = 1
Thread(4,0): 0 × 5 + 4 = 4
Thread(0,1): 1 × 5 + 0 = 5   ← 换行了
Thread(1,1): 1 × 5 + 1 = 6
Thread(0,2): 2 × 5 + 0 = 10  ← 又换行
~~~

**网格中的Block索引（一维化）**：

cpp

~~~cpp
int bid = blockIdx.y * gridDim.x + blockIdx.x;
```

**图解说明**（2×2网格）：
```
Block(0,0): 0 × 2 + 0 = 0
Block(1,0): 0 × 2 + 1 = 1
Block(0,1): 1 × 2 + 0 = 2
Block(1,1): 1 × 2 + 1 = 3
~~~

### 3.7 三维线程索引计算（完整公式）

**三维Block内的线程索引**：

cpp

~~~cpp
int tid = threadIdx.z * blockDim.x * blockDim.y 
        + threadIdx.y * blockDim.x 
        + threadIdx.x;
```

**记忆技巧**：从高维到低维，逐层展开
```
z维：每层有 blockDim.x × blockDim.y 个元素
y维：每行有 blockDim.x 个元素
x维：直接加上x坐标
~~~

**三维Grid中的Block索引**：

cpp

```cpp
int bid = blockIdx.z * gridDim.x * gridDim.y 
        + blockIdx.y * gridDim.x 
        + blockIdx.x;
```

### 3.8 实战：图像处理示例

**问题**：对1920×1080的图像进行灰度化处理

cpp

```cpp
__global__ void rgb_to_gray(unsigned char* rgb, unsigned char* gray, 
                           int width, int height) {
    // 计算当前线程处理的像素坐标
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // 边界检查
    if(x < width && y < height) {
        // 计算一维数组索引
        int idx = y * width + x;
        
        // RGB到灰度的转换
        unsigned char r = rgb[idx * 3 + 0];
        unsigned char g = rgb[idx * 3 + 1];
        unsigned char b = rgb[idx * 3 + 2];
        
        gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

int main() {
    int width = 1920, height = 1080;
    
    // 定义线程块：16×16 = 256个线程
    dim3 block_size(16, 16);
    
    // 计算需要多少个Block（向上取整）
    dim3 grid_size((width + 15) / 16,   // 1920/16 = 120个Block
                   (height + 15) / 16);  // 1080/16 = 68个Block
    
    // 启动核函数
    rgb_to_gray<<<grid_size, block_size>>>(d_rgb, d_gray, width, height);
}
```

**为什么使用16×16的Block？**

- 256个线程（16×16）是常用配置
- 正好是2的幂次，便于硬件优化
- 适合图像处理的二维结构

------

## 第四部分：网格和线程块的硬件限制

### 4.1 网格大小限制（重要！）

cpp

```cpp
gridDim.x 最大值: 2³¹ - 1 = 2,147,483,647
gridDim.y 最大值: 2¹⁶ - 1 = 65,535
gridDim.z 最大值: 2¹⁶ - 1 = 65,535
```

**为什么x维度特别大？**

- 一维配置最常用，给予更大的灵活性
- 可以处理非常大的数据集

**实例**：

cpp

```cpp
// ✅ 合法
kernel<<<2000000000, 256>>>();  // x维度很大

// ✅ 合法
dim3 grid(65535, 65535, 1);
kernel<<<grid, block>>>();

// ❌ 非法
dim3 grid(65536, 1, 1);  // y维度超限
```

### 4.2 线程块大小限制（关键约束！）

cpp

~~~cpp
blockDim.x 最大值: 1024
blockDim.y 最大值: 1024
blockDim.z 最大值: 64
```

**最关键的限制**：
```
blockDim.x × blockDim.y × blockDim.z ≤ 1024
~~~

⚠️ **这是最容易出错的地方！**

**合法配置示例**：

cpp

```cpp
// ✅ 正确
dim3 block1(1024, 1, 1);     // 1024 × 1 × 1 = 1024
dim3 block2(512, 2, 1);      // 512 × 2 × 1 = 1024
dim3 block3(32, 32, 1);      // 32 × 32 × 1 = 1024
dim3 block4(16, 16, 4);      // 16 × 16 × 4 = 1024
dim3 block5(8, 8, 16);       // 8 × 8 × 16 = 1024
```

**非法配置示例**：

cpp

```cpp
// ❌ 错误：总数超过1024
dim3 block1(32, 32, 2);      // 32 × 32 × 2 = 2048 > 1024
dim3 block2(1024, 2, 1);     // 1024 × 2 × 1 = 2048 > 1024

// ❌ 错误：单维度超限
dim3 block3(1, 1, 65);       // z维度 > 64
dim3 block4(2048, 1, 1);     // x维度 > 1024
```

### 4.3 为什么有这些限制？

**硬件原因**：

1. **寄存器数量有限**：每个SM（Streaming Multiprocessor）的寄存器数量固定
2. **共享内存有限**：线程块共享的内存大小有限（通常48KB或96KB）
3. **调度复杂度**：线程太多会增加调度开销

**1024的由来**：

- GPU以32个线程为一组（warp）调度
- 1024 = 32 × 32，可以组成32个warp
- 这是性能和资源的平衡点

### 4.4 选择合适的Block大小

**经验法则**：

1. **常用配置**：

cpp

```cpp
   128, 256, 512, 1024  // 一维
   16×16, 32×32         // 二维
   8×8×16              // 三维
```

1. **应该是32的倍数**（warp大小）：

cpp

```cpp
   // ✅ 推荐
   kernel<<<grid, 256>>>();  // 256 = 32 × 8
   
   // ⚠️ 不推荐（浪费资源）
   kernel<<<grid, 100>>>();  // 100不是32的倍数
```

1. 考虑寄存器使用

   ：

   - 如果核函数使用很多寄存器，减小Block大小
   - 使用`--ptxas-options=-v`编译选项查看寄存器使用情况

2. 考虑共享内存

   ：

   - 如果使用大量共享内存，减小Block大小

------

## 🔬 深度理解：物理执行模型

### Warp的概念

虽然课件没有详细讲，但理解warp对掌握线程模型很重要：

**什么是Warp？**

- GPU调度的基本单位
- **32个连续线程**组成一个warp
- 一个warp内的线程执行相同的指令（SIMT模型）

**示例**：

cpp

```cpp
kernel<<<1, 128>>>();

// 物理上分成4个warp：
// Warp 0: Thread 0-31
// Warp 1: Thread 32-63
// Warp 2: Thread 64-95
// Warp 3: Thread 96-127
```

**为什么重要？**

cpp

```cpp
__global__ void divergent_kernel() {
    if(threadIdx.x % 2 == 0) {
        // 分支A：偶数线程
    } else {
        // 分支B：奇数线程
    }
}
```

在一个warp内，有16个线程执行分支A，16个执行分支B，导致：

- 先执行分支A（另一半等待）
- 再执行分支B（前一半等待）
- 性能降低50%！这叫**warp分歧**

------

## 💡 实战技巧和最佳实践

### 技巧1：计算Grid大小的通用公式

cpp

~~~cpp
// 处理n个元素，每个Block有block_size个线程
int grid_size = (n + block_size - 1) / block_size;

// 或者使用宏（更清晰）
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
int grid_size = CEIL_DIV(n, block_size);
```

**为什么要向上取整？**
```
例如：n = 1000, block_size = 256

方案1（错误）：1000 / 256 = 3（整数除法）
3 × 256 = 768，只处理768个元素，丢失232个！

方案2（正确）：(1000 + 255) / 256 = 4
4 × 256 = 1024，能处理所有1000个元素
~~~

### 技巧2：二维数据的线程配置

cpp

```cpp
// 处理width×height的二维数据
dim3 block(16, 16);  // 每个Block 16×16 = 256个线程

dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);

// 核函数中：
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if(x < width && y < height) {
    // 处理(x, y)位置的数据
}
```

### 技巧3：边界检查的重要性

cpp

~~~cpp
__global__ void process(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // ⚠️ 必须检查边界！
    if(idx < n) {
        data[idx] = data[idx] * 2;
    }
    // 没有if会导致越界访问，程序崩溃！
}
```

**为什么需要？**
```
例如：n = 1000, 配置<<<4, 256>>>

实际启动 4 × 256 = 1024 个线程
但只有1000个数据
最后24个线程(idx = 1000-1023)会越界访问！
~~~

### 技巧4：性能调优

**实验找到最佳Block大小**：

cpp

```cpp
void benchmark_block_size() {
    int sizes[] = {64, 128, 256, 512, 1024};
    
    for(int i = 0; i < 5; i++) {
        int block_size = sizes[i];
        int grid_size = (n + block_size - 1) / block_size;
        
        // 计时
        auto start = chrono::high_resolution_clock::now();
        kernel<<<grid_size, block_size>>>(data, n);
        cudaDeviceSynchronize();
        auto end = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        printf("Block size %d: %ld us\n", block_size, duration.count());
    }
}
```

------

## 📊 完整示例：矩阵加法

让我们用一个完整的例子把所有知识串起来：

cpp

```cpp
#include <stdio.h>

// 矩阵加法核函数（二维线程）
__global__ void matrix_add(float* A, float* B, float* C, 
                          int width, int height) {
    // 计算当前线程处理的元素坐标
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    // 边界检查
    if(col < width && row < height) {
        // 二维坐标转一维索引
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // 矩阵大小
    int width = 1024, height = 1024;
    int size = width * height * sizeof(float);
    
    // 1. 分配主机内存
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
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
    dim3 block_size(16, 16);  // 256个线程per block
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n", 
           grid_size.x, grid_size.y, block_size.x, block_size.y);
    printf("Total threads: %d\n", 
           grid_size.x * grid_size.y * block_size.x * block_size.y);
    
    // 6. 启动核函数
    matrix_add<<<grid_size, block_size>>>(d_A, d_B, d_C, width, height);
    
    // 7. 拷贝结果回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 8. 验证结果
    bool correct = true;
    for(int i = 0; i < 10; i++) {
        printf("C[%d] = %.0f (expected %.0f)\n", 
               i, h_C[i], h_A[i] + h_B[i]);
        if(h_C[i] != h_A[i] + h_B[i]) correct = false;
    }
    printf(correct ? "Result correct!\n" : "Result wrong!\n");
    
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

**编译运行**：

bash

~~~bash
nvcc matrix_add.cu -o matrix_add
./matrix_add
```

**输出示例**：
```
Grid: (64, 64), Block: (16, 16)
Total threads: 1048576
C[0] = 0 (expected 0)
C[1] = 3 (expected 3)
...
Result correct!
~~~

------

## 🎓 学习检查清单

完成本节学习后，你应该能够：

-  理解Grid、Block、Thread的层次结构
-  知道线程分块是逻辑划分，物理不分块
-  掌握内建变量的使用（gridDim、blockDim、blockIdx、threadIdx）
-  计算一维线程的全局ID
-  配置二维和三维的Grid和Block
-  计算多维线程的索引
-  记住1024的线程块大小限制
-  知道如何选择合适的Block大小
-  理解为什么需要边界检查

------

通过本节课的学习，您已经掌握了CUDA线程模型的核心概念。这是CUDA编程的基础，接下来您将学习内存管理、同步机制等更高级的主题。记住：理解线程模型是写出高效CUDA程序的关键！





