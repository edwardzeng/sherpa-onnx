# C++全面学习指南 - 基于sherpa-onnx项目（第二部分）

## 第二部分：面向对象编程

## 类和对象

### 1. 类的基本定义

```cpp
// 基本类定义
class Wave {
public:
    std::vector<float> samples;
    int32_t sample_rate;
};

// 更完整的类定义（来自sherpa-onnx）
class CircularBuffer {
public:
    // 构造函数
    explicit CircularBuffer(int32_t capacity);
    
    // 成员函数
    void Push(const float *p, int32_t n);
    std::vector<float> Get(int32_t start_index, int32_t n) const;
    void Pop(int32_t n);
    
    // 内联成员函数
    int32_t Size() const { return tail_ - head_; }
    int32_t Head() const { return head_; }
    int32_t Tail() const { return tail_; }
    
    void Reset() {
        head_ = 0;
        tail_ = 0;
    }

private:
    // 私有成员变量
    std::vector<float> buffer_;
    int32_t head_ = 0;
    int32_t tail_ = 0;
};
```

### 2. 访问控制

```cpp
class AudioProcessor {
public:
    // 公有成员：对外接口
    void ProcessAudio(const float* input, float* output, int32_t n);
    float GetGain() const { return gain_; }
    void SetGain(float gain) { gain_ = gain; }

protected:
    // 保护成员：子类可访问
    virtual void ApplyEffect(float* samples, int32_t n);
    float gain_ = 1.0f;

private:
    // 私有成员：仅类内部可访问
    void NormalizeAudio(float* samples, int32_t n);
    std::vector<float> internal_buffer_;
};
```

### 3. this指针

```cpp
class OnlineStream {
private:
    int32_t num_processed_frames_ = 0;
    
public:
    void Reset() {
        // this指针指向当前对象
        this->num_processed_frames_ = 0;
        // 可以省略this->
        num_processed_frames_ = 0;
    }
    
    // 返回自身引用，支持链式调用
    OnlineStream& Configure(int32_t frames) {
        num_processed_frames_ = frames;
        return *this;
    }
};

// 链式调用
OnlineStream stream;
stream.Configure(100).Reset();
```

### 4. 静态成员

```cpp
class ModelManager {
private:
    static int32_t instance_count_;  // 静态成员变量
    std::string model_name_;
    
public:
    ModelManager() {
        ++instance_count_;
    }
    
    ~ModelManager() {
        --instance_count_;
    }
    
    // 静态成员函数
    static int32_t GetInstanceCount() {
        return instance_count_;
    }
    
    // 静态工厂方法
    static std::unique_ptr<ModelManager> Create(const std::string& name) {
        return std::make_unique<ModelManager>();
    }
};

// 静态成员变量定义（在.cpp文件中）
int32_t ModelManager::instance_count_ = 0;
```

### 5. 结构体 vs 类

```cpp
// 结构体（默认public）
struct Point {
    float x;
    float y;
    
    float Distance() const {
        return std::sqrt(x * x + y * y);
    }
};

// 类（默认private）
class Point2D {
    float x_;  // 默认private
    float y_;
    
public:
    Point2D(float x, float y) : x_(x), y_(y) {}
    float GetX() const { return x_; }
    float GetY() const { return y_; }
};

// sherpa-onnx中的结构体示例
struct OnlineRecognizerResult {
    std::string text;
    std::vector<std::string> tokens;
    std::vector<float> timestamps;
    int32_t segment = 0;
    float start_time = 0;
    bool is_final = false;
    
    // 结构体也可以有成员函数
    std::string AsJsonString() const;
};
```

## 构造函数和析构函数

### 1. 默认构造函数

```cpp
class AudioBuffer {
private:
    float* data_;
    size_t size_;
    
public:
    // 默认构造函数
    AudioBuffer() : data_(nullptr), size_(0) {}
    
    // 带参数的构造函数
    explicit AudioBuffer(size_t size) 
        : data_(new float[size]), size_(size) {
        std::fill(data_, data_ + size_, 0.0f);
    }
};
```

### 2. 初始化列表

```cpp
class FeatureExtractor {
private:
    const int32_t sample_rate_;    // const成员必须在初始化列表中初始化
    int32_t feature_dim_;
    bool normalize_;
    
public:
    // 使用初始化列表
    FeatureExtractor(int32_t sample_rate, int32_t dim, bool norm)
        : sample_rate_(sample_rate),  // 初始化const成员
          feature_dim_(dim),
          normalize_(norm) {
        // 构造函数体
    }
    
    // 委托构造函数（C++11）
    FeatureExtractor() : FeatureExtractor(16000, 80, true) {}
};
```

### 3. 拷贝构造函数

```cpp
class Buffer {
private:
    float* data_;
    size_t size_;
    
public:
    // 拷贝构造函数
    Buffer(const Buffer& other) 
        : size_(other.size_), data_(new float[size_]) {
        std::copy(other.data_, other.data_ + size_, data_);
    }
    
    // 禁用拷贝构造
    // Buffer(const Buffer&) = delete;
};

// 深拷贝 vs 浅拷贝
class ShallowCopy {
    int* ptr_;
public:
    ShallowCopy(const ShallowCopy& other) : ptr_(other.ptr_) {}
    // 危险！两个对象共享同一个指针
};

class DeepCopy {
    int* ptr_;
public:
    DeepCopy(const DeepCopy& other) 
        : ptr_(new int(*other.ptr_)) {}
    // 安全！每个对象有自己的副本
};
```

### 4. 移动构造函数（C++11）

```cpp
class OnlineStream::Impl {
private:
    std::vector<Ort::Value> states_;
    FeatureExtractor feat_extractor_;
    
public:
    // 移动构造函数
    Impl(Impl&& other) noexcept
        : states_(std::move(other.states_)),
          feat_extractor_(std::move(other.feat_extractor_)) {
        // other对象的资源被"移动"到this
    }
    
    // 移动赋值运算符
    Impl& operator=(Impl&& other) noexcept {
        if (this != &other) {
            states_ = std::move(other.states_);
            feat_extractor_ = std::move(other.feat_extractor_);
        }
        return *this;
    }
};
```

### 5. 析构函数

```cpp
class ResourceManager {
private:
    FILE* file_;
    float* buffer_;
    std::thread* worker_;
    
public:
    ResourceManager() 
        : file_(nullptr), buffer_(nullptr), worker_(nullptr) {}
    
    // 析构函数
    ~ResourceManager() {
        // 释放资源的顺序很重要
        if (worker_) {
            worker_->join();
            delete worker_;
        }
        
        if (buffer_) {
            delete[] buffer_;
        }
        
        if (file_) {
            fclose(file_);
        }
    }
};

// 虚析构函数（用于多态基类）
class Base {
public:
    virtual ~Base() = default;  // 虚析构函数
};

class Derived : public Base {
private:
    int* data_;
public:
    ~Derived() {
        delete[] data_;
    }
};
```

## 继承和多态

### 1. 基本继承

```cpp
// 基类
class VadModel {
public:
    virtual ~VadModel() = default;
    
    // 纯虚函数
    virtual void Reset() = 0;
    virtual bool IsSpeech(const float *samples, int32_t n) = 0;
    virtual int32_t WindowSize() const = 0;
    
protected:
    // 保护成员，子类可访问
    float threshold_ = 0.5f;
};

// 派生类
class SileroVadModel : public VadModel {
private:
    std::unique_ptr<Model> model_;
    
public:
    // 实现纯虚函数
    void Reset() override {
        // 重置状态
    }
    
    bool IsSpeech(const float *samples, int32_t n) override {
        // 实现语音检测逻辑
        return true;
    }
    
    int32_t WindowSize() const override {
        return 512;
    }
};
```

### 2. 多重继承

```cpp
// 接口类（纯虚类）
class IAudioInput {
public:
    virtual ~IAudioInput() = default;
    virtual void ReadAudio(float* buffer, int32_t n) = 0;
};

class IAudioOutput {
public:
    virtual ~IAudioOutput() = default;
    virtual void WriteAudio(const float* buffer, int32_t n) = 0;
};

// 多重继承
class AudioDevice : public IAudioInput, public IAudioOutput {
public:
    void ReadAudio(float* buffer, int32_t n) override {
        // 实现音频输入
    }
    
    void WriteAudio(const float* buffer, int32_t n) override {
        // 实现音频输出
    }
};
```

### 3. 虚函数和动态绑定

```cpp
// 基类
class SpeakerEmbeddingExtractorImpl {
public:
    virtual ~SpeakerEmbeddingExtractorImpl() = default;
    
    // 虚函数
    virtual int32_t Dim() const = 0;
    virtual std::unique_ptr<OnlineStream> CreateStream() const = 0;
    virtual bool IsReady(OnlineStream *s) const = 0;
    virtual std::vector<float> Compute(OnlineStream *s) const = 0;
};

// 派生类
class SpeakerEmbeddingExtractorNeMoImpl 
    : public SpeakerEmbeddingExtractorImpl {
public:
    int32_t Dim() const override {
        return model_.GetMetaData().output_dim;
    }
    
    std::unique_ptr<OnlineStream> CreateStream() const override {
        return std::make_unique<OnlineStream>(feat_config_);
    }
    
    // 其他虚函数实现...
};

// 多态使用
void ProcessEmbedding(SpeakerEmbeddingExtractorImpl* extractor,
                     OnlineStream* stream) {
    if (extractor->IsReady(stream)) {
        auto embedding = extractor->Compute(stream);
        // 处理embedding...
    }
}
```

### 4. 抽象类和接口

```cpp
// 抽象类（含有纯虚函数）
class Decoder {
public:
    virtual ~Decoder() = default;
    
    // 纯虚函数
    virtual void Decode(const float* input, int32_t n) = 0;
    virtual std::string GetResult() const = 0;
    
    // 普通虚函数（提供默认实现）
    virtual void Reset() {
        // 默认重置行为
    }
};

// 具体实现类
class TransducerDecoder : public Decoder {
private:
    std::string result_;
    
public:
    void Decode(const float* input, int32_t n) override {
        // 实现解码逻辑
    }
    
    std::string GetResult() const override {
        return result_;
    }
};
```

### 5. 虚函数表和性能

```cpp
// 虚函数的开销
class Base {
public:
    virtual void VirtualFunc() {}  // 有虚函数表开销
    void NonVirtualFunc() {}        // 无开销，可内联
};

// final关键字（C++11）
class FinalClass final {  // 不能被继承
public:
    virtual void Func() final;  // 不能被重写
};

// 虚函数最佳实践
class PerformanceOptimized {
public:
    // 仅在需要多态时使用虚函数
    virtual ~PerformanceOptimized() = default;
    
    // 频繁调用的函数避免使用虚函数
    inline void FastOperation() {
        // 内联实现
    }
    
    // 需要多态的函数使用虚函数
    virtual void PolymorphicOperation() = 0;
};
```

## 运算符重载

### 1. 基本运算符重载

```cpp
class Complex {
private:
    double real_;
    double imag_;
    
public:
    Complex(double r = 0, double i = 0) : real_(r), imag_(i) {}
    
    // 加法运算符重载（成员函数）
    Complex operator+(const Complex& other) const {
        return Complex(real_ + other.real_, imag_ + other.imag_);
    }
    
    // 减法运算符重载（友元函数）
    friend Complex operator-(const Complex& lhs, const Complex& rhs) {
        return Complex(lhs.real_ - rhs.real_, lhs.imag_ - rhs.imag_);
    }
    
    // 复合赋值运算符
    Complex& operator+=(const Complex& other) {
        real_ += other.real_;
        imag_ += other.imag_;
        return *this;
    }
    
    // 相等运算符
    bool operator==(const Complex& other) const {
        return real_ == other.real_ && imag_ == other.imag_;
    }
};
```

### 2. 输入输出运算符重载

```cpp
class Point {
private:
    float x_, y_;
    
public:
    Point(float x, float y) : x_(x), y_(y) {}
    
    // 输出运算符（必须是友元或非成员函数）
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "(" << p.x_ << ", " << p.y_ << ")";
        return os;
    }
    
    // 输入运算符
    friend std::istream& operator>>(std::istream& is, Point& p) {
        is >> p.x_ >> p.y_;
        return is;
    }
};

// 使用
Point p(3.14, 2.71);
std::cout << "Point: " << p << std::endl;
```

### 3. 下标运算符重载

```cpp
// 来自sherpa-onnx的SymbolTable类
class SymbolTable {
private:
    std::unordered_map<std::string, int32_t> sym2id_;
    std::unordered_map<int32_t, std::string> id2sym_;
    
public:
    // 下标运算符重载（const版本）
    const std::string operator[](int32_t id) const {
        auto it = id2sym_.find(id);
        if (it != id2sym_.end()) {
            return it->second;
        }
        return "";
    }
    
    // 下标运算符重载（通过符号查找ID）
    int32_t operator[](const std::string &sym) const {
        auto it = sym2id_.find(sym);
        if (it != sym2id_.end()) {
            return it->second;
        }
        return -1;
    }
};
```

### 4. 函数调用运算符重载

```cpp
// 函数对象（仿函数）
class AudioProcessor {
private:
    float gain_;
    
public:
    explicit AudioProcessor(float gain) : gain_(gain) {}
    
    // 函数调用运算符
    void operator()(float* samples, int32_t n) const {
        for (int32_t i = 0; i < n; ++i) {
            samples[i] *= gain_;
        }
    }
};

// 使用
AudioProcessor processor(0.5f);
float samples[100];
processor(samples, 100);  // 像函数一样调用

// Lambda表达式的等价实现
auto lambda_processor = [gain = 0.5f](float* samples, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        samples[i] *= gain;
    }
};
```

### 5. 类型转换运算符

```cpp
class Fraction {
private:
    int numerator_;
    int denominator_;
    
public:
    Fraction(int n, int d) : numerator_(n), denominator_(d) {}
    
    // 转换为double
    operator double() const {
        return static_cast<double>(numerator_) / denominator_;
    }
    
    // explicit防止隐式转换
    explicit operator int() const {
        return numerator_ / denominator_;
    }
};

// 使用
Fraction f(3, 4);
double d = f;              // 隐式转换为double
int i = static_cast<int>(f); // 需要显式转换
```

## 友元函数和友元类

### 1. 友元函数

```cpp
class Vector3D {
private:
    float x_, y_, z_;
    
public:
    Vector3D(float x, float y, float z) : x_(x), y_(y), z_(z) {}
    
    // 声明友元函数
    friend float DotProduct(const Vector3D& a, const Vector3D& b);
    friend std::ostream& operator<<(std::ostream& os, const Vector3D& v);
};

// 友元函数定义（可以访问私有成员）
float DotProduct(const Vector3D& a, const Vector3D& b) {
    return a.x_ * b.x_ + a.y_ * b.y_ + a.z_ * b.z_;
}

std::ostream& operator<<(std::ostream& os, const Vector3D& v) {
    os << "(" << v.x_ << ", " << v.y_ << ", " << v.z_ << ")";
    return os;
}
```

### 2. 友元类

```cpp
class AudioEngine;  // 前向声明

class AudioBuffer {
private:
    float* data_;
    size_t size_;
    
    // AudioEngine可以访问AudioBuffer的私有成员
    friend class AudioEngine;
    
public:
    explicit AudioBuffer(size_t size) 
        : data_(new float[size]), size_(size) {}
};

class AudioEngine {
public:
    void ProcessBuffer(AudioBuffer& buffer) {
        // 可以直接访问buffer的私有成员
        for (size_t i = 0; i < buffer.size_; ++i) {
            buffer.data_[i] *= 0.5f;
        }
    }
};
```

### 3. 友元的使用原则

```cpp
// 好的设计：最小化友元使用
class BankAccount {
private:
    double balance_;
    
public:
    // 提供公共接口而不是友元
    double GetBalance() const { return balance_; }
    void Deposit(double amount) { balance_ += amount; }
    bool Withdraw(double amount) {
        if (amount <= balance_) {
            balance_ -= amount;
            return true;
        }
        return false;
    }
};

// 避免过度使用友元
class BadDesign {
private:
    int data_;
    // 太多友元破坏了封装
    friend class Class1;
    friend class Class2;
    friend class Class3;
    // ...
};
```

### 4. 友元模板

```cpp
template <typename T>
class Matrix {
private:
    T* data_;
    int rows_, cols_;
    
public:
    // 友元函数模板
    template <typename U>
    friend Matrix<U> operator*(const Matrix<U>& a, const Matrix<U>& b);
    
    // 特定实例化的友元
    friend class MatrixOperations<T>;
};

template <typename T>
Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b) {
    // 可以访问Matrix<T>的私有成员
    // 实现矩阵乘法
}
```

---

## 小结

本部分详细介绍了C++的面向对象编程特性：

1. **类和对象**：类定义、访问控制、this指针、静态成员
2. **构造函数和析构函数**：各种构造函数、初始化列表、RAII
3. **继承和多态**：继承机制、虚函数、抽象类、动态绑定
4. **运算符重载**：各种运算符的重载方法和使用场景
5. **友元函数和友元类**：友元的概念、使用和设计原则

这些面向对象的特性是C++强大功能的核心，掌握它们对于设计和实现复杂的软件系统至关重要。在下一部分中，我们将探讨C++的高级特性，包括模板编程、STL容器、智能指针等。