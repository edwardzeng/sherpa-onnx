# C++全面学习指南 - 基于sherpa-onnx项目（第一部分）

本文档是一个详细的C++学习指南，通过分析sherpa-onnx项目的实际代码，帮助您系统地学习C++编程语言。

## 目录

### 第一部分：C++基础语法
1. [变量和基本数据类型](#变量和基本数据类型)
2. [数组和指针](#数组和指针)
3. [引用](#引用)
4. [函数](#函数)
5. [内存管理](#内存管理)

### 第二部分：面向对象编程
6. [类和对象](#类和对象)
7. [构造函数和析构函数](#构造函数和析构函数)
8. [继承和多态](#继承和多态)
9. [运算符重载](#运算符重载)
10. [友元函数和友元类](#友元函数和友元类)

### 第三部分：高级特性
11. [模板编程](#模板编程)
12. [STL容器](#stl容器)
13. [智能指针](#智能指针)
14. [异常处理](#异常处理)
15. [并发编程](#并发编程)

---

## 第一部分：C++基础语法

## 变量和基本数据类型

### 1. 基本数据类型

```cpp
// 整数类型
int32_t sample_rate = 16000;          // 32位有符号整数
uint32_t buffer_size = 1024;          // 32位无符号整数
int64_t timestamp = 1234567890;       // 64位有符号整数
size_t array_size = 100;              // 与平台相关的无符号整数

// 浮点类型
float confidence = 0.95f;             // 单精度浮点数
double precision = 0.123456789;       // 双精度浮点数

// 字符类型
char letter = 'A';                    // 单个字符
char* c_string = "Hello";             // C风格字符串
std::string cpp_string = "World";     // C++字符串

// 布尔类型
bool is_valid = true;
bool is_empty = false;
```

### 2. 常量和字面量

```cpp
// 常量定义
const int32_t MAX_FRAMES = 1000;
const float PI = 3.14159f;
const char* const MODEL_TYPE = "transducer";

// constexpr（编译时常量）
constexpr int32_t BUFFER_SIZE = 256;
constexpr float EPSILON = 1e-6f;

// 宏定义（来自sherpa-onnx）
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif
```

### 3. 类型推导和auto

```cpp
// auto关键字自动推导类型
auto value = 42;                      // int
auto ratio = 3.14;                    // double
auto name = std::string("sherpa");    // std::string

// 在循环中使用auto
std::vector<int> numbers = {1, 2, 3, 4, 5};
for (auto num : numbers) {
    std::cout << num << " ";
}

// 使用auto处理复杂类型
auto result = recognizer.GetResult(&stream);  // 自动推导返回类型
```

## 数组和指针

### 1. 静态数组

```cpp
// 一维数组
int32_t samples[1024];                // 声明
float coefficients[5] = {0.1, 0.2, 0.3, 0.4, 0.5};  // 初始化

// 多维数组
float matrix[3][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f}
};

// std::array（C++11）
#include <array>
std::array<float, 8000> tail_paddings = {0};  // 初始化为0
std::array<int64_t, 3> shape{1, num_frames, feat_dim};
```

### 2. 动态数组（向量）

```cpp
// std::vector的使用
std::vector<float> samples;           // 空向量
std::vector<int32_t> tokens(100);     // 100个元素，初始化为0
std::vector<std::string> words = {"hello", "world"};  // 初始化列表

// 向量操作
samples.push_back(0.5f);              // 添加元素
samples.resize(1000);                 // 调整大小
samples.clear();                      // 清空

// 访问元素
float first = samples[0];             // 下标访问（不检查边界）
float second = samples.at(1);         // at访问（检查边界）
float* data = samples.data();         // 获取底层数组指针
```

### 3. 指针基础

```cpp
// 指针声明和初始化
int value = 42;
int* ptr = &value;                    // 指向value的指针
int* null_ptr = nullptr;              // 空指针（C++11）

// 指针操作
*ptr = 100;                           // 解引用，修改value的值
int result = *ptr;                    // 读取指针指向的值

// 指针运算
float samples[100];
float* p = samples;                   // 指向数组首元素
p++;                                  // 指向下一个元素
float third = *(p + 2);               // 访问第三个元素

// 函数指针
typedef void (*Callback)(const float*, int32_t);
Callback process_audio = nullptr;
```

### 4. 指针和数组的关系

```cpp
// 数组名作为指针
void ProcessSamples(const float* samples, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        float sample = samples[i];    // 等价于 *(samples + i)
        // 处理样本...
    }
}

// 实际使用示例（来自sherpa-onnx）
void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                    int32_t n) const {
    // waveform是指向音频数据的指针
    // n是音频样本的数量
}
```

### 5. 多级指针

```cpp
// 二级指针
int value = 42;
int* ptr = &value;
int** ptr_to_ptr = &ptr;

// 指针数组
const char* keywords[] = {"hello", "world", "sherpa"};

// 动态二维数组
float** matrix = new float*[rows];
for (int i = 0; i < rows; ++i) {
    matrix[i] = new float[cols];
}

// 记得释放内存
for (int i = 0; i < rows; ++i) {
    delete[] matrix[i];
}
delete[] matrix;
```

## 引用

### 1. 引用基础

```cpp
// 引用声明
int value = 42;
int& ref = value;                     // ref是value的引用
ref = 100;                            // 修改value的值

// 常量引用
const int& const_ref = value;         // 不能通过const_ref修改value

// 引用必须初始化
// int& invalid_ref;                  // 错误！引用必须初始化
```

### 2. 函数参数中的引用

```cpp
// 传值 vs 传引用
void ModifyByValue(int x) {
    x = 100;  // 只修改局部副本
}

void ModifyByReference(int& x) {
    x = 100;  // 修改原始值
}

// 常量引用参数（避免拷贝，提高效率）
void ProcessConfig(const OnlineRecognizerConfig& config) {
    // 可以读取config，但不能修改
    int32_t threads = config.model_config.num_threads;
}

// 返回引用
int32_t& GetNumProcessedFrames() {
    return num_processed_frames_;
}
```

### 3. 引用 vs 指针

```cpp
// 引用的特点：
// 1. 必须初始化
// 2. 不能重新绑定
// 3. 没有空引用
// 4. 使用时不需要解引用

// 指针的特点：
// 1. 可以不初始化
// 2. 可以重新赋值
// 3. 可以为nullptr
// 4. 需要解引用才能访问值

// 何时使用引用
void Swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

// 何时使用指针
void ProcessOptional(const Config* config) {
    if (config != nullptr) {
        // 处理配置
    }
}
```

### 4. 右值引用和移动语义（C++11）

```cpp
// 左值和右值
int x = 42;                           // x是左值，42是右值
int& lref = x;                        // 左值引用
// int& invalid = 42;                 // 错误！不能绑定到右值

// 右值引用
int&& rref = 42;                      // 右值引用可以绑定到右值
int&& moved = std::move(x);           // std::move将左值转为右值

// 移动构造函数
class Buffer {
private:
    float* data_;
    size_t size_;
    
public:
    // 移动构造函数
    Buffer(Buffer&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    // 移动赋值运算符
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
};
```

## 函数

### 1. 函数基础

```cpp
// 函数声明
int Add(int a, int b);                // 声明
float Calculate(float x, float y);     // 声明

// 函数定义
int Add(int a, int b) {
    return a + b;
}

// 内联函数
inline float Square(float x) {
    return x * x;
}

// 带默认参数的函数
void ProcessAudio(const float* samples, int32_t n, 
                  float gain = 1.0f) {
    for (int32_t i = 0; i < n; ++i) {
        samples[i] *= gain;
    }
}
```

### 2. 函数重载

```cpp
// 同名函数，不同参数
float Dot(const float* a, const float* b, int n);
double Dot(const double* a, const double* b, int n);

// 实际例子（来自sherpa-onnx）
Wave ReadWave(const std::string& filename);
Wave ReadWave(std::istream& is);
```

### 3. 函数模板

```cpp
// 函数模板定义
template <typename T>
T Max(T a, T b) {
    return (a > b) ? a : b;
}

// 使用函数模板
int max_int = Max(10, 20);           // T = int
float max_float = Max(3.14f, 2.71f); // T = float

// 实际例子（来自sherpa-onnx）
template <class I>
static I Gcd(I m, I n) {
    static_assert(std::is_integral_v<I>);
    while (true) {
        m %= n;
        if (m == 0) return (n > 0 ? n : -n);
        n %= m;
        if (n == 0) return (m > 0 ? m : -m);
    }
}
```

### 4. Lambda表达式（C++11）

```cpp
// 基本lambda
auto add = [](int a, int b) { return a + b; };
int result = add(3, 4);  // result = 7

// 捕获变量
int multiplier = 10;
auto multiply = [multiplier](int x) { return x * multiplier; };

// 捕获方式
int x = 10, y = 20;
auto f1 = [x, y]() { return x + y; };      // 按值捕获
auto f2 = [&x, &y]() { x++; y++; };        // 按引用捕获
auto f3 = [=]() { return x + y; };         // 按值捕获所有
auto f4 = [&]() { x++; y++; };             // 按引用捕获所有

// 在STL算法中使用
std::vector<int> nums = {1, 2, 3, 4, 5};
std::for_each(nums.begin(), nums.end(), 
              [](int& n) { n *= 2; });
```

## 内存管理

### 1. 栈内存 vs 堆内存

```cpp
// 栈内存（自动管理）
void StackMemory() {
    int local_var = 42;               // 栈上分配
    float array[100];                 // 栈上分配
    // 函数结束时自动释放
}

// 堆内存（手动管理）
void HeapMemory() {
    int* ptr = new int(42);           // 堆上分配
    float* array = new float[100];    // 堆上分配数组
    
    // 使用内存...
    
    delete ptr;                       // 释放单个对象
    delete[] array;                   // 释放数组
}
```

### 2. 动态内存分配

```cpp
// new和delete
class AudioBuffer {
private:
    float* data_;
    size_t size_;
    
public:
    AudioBuffer(size_t size) : size_(size) {
        data_ = new float[size_];
        std::fill(data_, data_ + size_, 0.0f);
    }
    
    ~AudioBuffer() {
        delete[] data_;
    }
    
    // 防止浅拷贝
    AudioBuffer(const AudioBuffer&) = delete;
    AudioBuffer& operator=(const AudioBuffer&) = delete;
};

// placement new（在指定位置构造对象）
char buffer[sizeof(AudioBuffer)];
AudioBuffer* ab = new (buffer) AudioBuffer(1024);
ab->~AudioBuffer();  // 需要显式调用析构函数
```

### 3. 内存泄漏和常见问题

```cpp
// 内存泄漏示例
void MemoryLeak() {
    int* ptr = new int[100];
    // 忘记delete[] ptr;
    // 内存泄漏！
}

// 双重释放
void DoubleFree() {
    int* ptr = new int(42);
    delete ptr;
    // delete ptr;  // 错误！双重释放
}

// 使用已释放的内存
void UseAfterFree() {
    int* ptr = new int(42);
    delete ptr;
    // *ptr = 100;  // 错误！使用已释放的内存
}

// 数组越界
void ArrayOverflow() {
    int array[10];
    // array[10] = 42;  // 错误！越界访问
}
```

### 4. RAII（资源获取即初始化）

```cpp
// RAII示例
class FileHandle {
private:
    FILE* file_;
    
public:
    explicit FileHandle(const std::string& filename) {
        file_ = fopen(filename.c_str(), "r");
        if (!file_) {
            throw std::runtime_error("Failed to open file");
        }
    }
    
    ~FileHandle() {
        if (file_) {
            fclose(file_);
        }
    }
    
    // 禁用拷贝
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    
    // 启用移动
    FileHandle(FileHandle&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }
    
    FILE* get() const { return file_; }
};

// 使用RAII
void ProcessFile(const std::string& filename) {
    FileHandle file(filename);
    // 使用file...
    // 析构函数自动关闭文件
}
```

### 5. 内存对齐

```cpp
// 对齐说明符（C++11）
struct alignas(16) AlignedData {
    float data[4];
};

// 检查对齐
size_t alignment = alignof(AlignedData);

// 手动对齐分配
void* AlignedAlloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
    #ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
    #else
        posix_memalign(&ptr, alignment, size);
    #endif
    return ptr;
}
```

---

## 小结

本部分介绍了C++的基础语法，包括：

1. **变量和数据类型**：基本类型、常量、类型推导
2. **数组和指针**：静态数组、动态数组、指针操作、指针运算
3. **引用**：左值引用、右值引用、引用参数、移动语义
4. **函数**：函数定义、重载、模板、Lambda表达式
5. **内存管理**：栈内存、堆内存、动态分配、RAII

这些是C++编程的基础，理解这些概念对于阅读和编写C++代码至关重要。在下一部分中，我们将深入探讨面向对象编程的概念。