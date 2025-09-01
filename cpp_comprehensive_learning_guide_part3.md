# C++全面学习指南 - 基于sherpa-onnx项目（第三部分）

## 第三部分：高级特性

## 模板编程

### 1. 函数模板

```cpp
// 基本函数模板
template <typename T>
T Max(T a, T b) {
    return (a > b) ? a : b;
}

// 多个模板参数
template <typename T, typename U>
auto Add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// 来自sherpa-onnx的例子：最大公约数
template <class I>
static I Gcd(I m, I n) {
    static_assert(std::is_integral<I>::value, "");
    if (m == 0 || n == 0) {
        if (m == 0 && n == 0) {
            fprintf(stderr, "Undefined GCD since m = 0, n = 0.\n");
            exit(-1);
        }
        return (m == 0 ? (n > 0 ? n : -n) : (m > 0 ? m : -m));
    }
    while (true) {
        m %= n;
        if (m == 0) return (n > 0 ? n : -n);
        n %= m;
        if (n == 0) return (m > 0 ? m : -m);
    }
}

// 模板特化
template <typename T>
void Print(T value) {
    std::cout << value << std::endl;
}

// 特化版本for const char*
template <>
void Print<const char*>(const char* value) {
    std::cout << "String: " << value << std::endl;
}
```

### 2. 类模板

```cpp
// 基本类模板
template <typename T>
class Stack {
private:
    std::vector<T> elements_;
    
public:
    void Push(const T& elem) {
        elements_.push_back(elem);
    }
    
    void Push(T&& elem) {
        elements_.push_back(std::move(elem));
    }
    
    T Pop() {
        if (elements_.empty()) {
            throw std::runtime_error("Stack is empty");
        }
        T elem = std::move(elements_.back());
        elements_.pop_back();
        return elem;
    }
    
    bool Empty() const {
        return elements_.empty();
    }
    
    size_t Size() const {
        return elements_.size();
    }
};

// 使用类模板
Stack<int> intStack;
intStack.Push(42);
Stack<std::string> stringStack;
stringStack.Push("Hello");
```

### 3. 模板元编程

```cpp
// 编译时计算阶乘
template <int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

// 特化终止递归
template <>
struct Factorial<0> {
    static constexpr int value = 1;
};

// 使用
constexpr int fact5 = Factorial<5>::value;  // 120

// 类型特征（Type Traits）
template <typename T>
struct IsPointer {
    static constexpr bool value = false;
};

template <typename T>
struct IsPointer<T*> {
    static constexpr bool value = true;
};

// SFINAE（Substitution Failure Is Not An Error）
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
SafeDivide(T a, T b) {
    if (b == 0) {
        throw std::runtime_error("Division by zero");
    }
    return a / b;
}
```

### 4. 可变参数模板（C++11）

```cpp
// 递归模板展开
template <typename T>
void Print(T&& t) {
    std::cout << t << std::endl;
}

template <typename T, typename... Args>
void Print(T&& t, Args&&... args) {
    std::cout << t << " ";
    Print(std::forward<Args>(args)...);
}

// 使用
Print("Hello", "World", 42, 3.14);  // 输出: Hello World 42 3.14

// 来自sherpa-onnx的例子：创建多维数组
template <typename T = float>
Ort::Value Stack(OrtAllocator *allocator,
                 const std::vector<const Ort::Value *> &values, 
                 int32_t dim);
```

### 5. 模板别名（C++11）

```cpp
// using创建模板别名
template <typename T>
using Vec = std::vector<T>;

template <typename K, typename V>
using Map = std::unordered_map<K, V>;

// 使用
Vec<int> numbers = {1, 2, 3, 4, 5};
Map<std::string, int> word_count;

// 来自sherpa-onnx的例子
using ContextGraphPtr = std::shared_ptr<ContextGraph>;
```

## STL容器

### 1. 序列容器

```cpp
// vector - 动态数组
std::vector<float> samples;
samples.reserve(1000);            // 预分配空间
samples.push_back(0.5f);          // 添加元素
samples.emplace_back(0.7f);       // 原地构造
samples.resize(100);              // 调整大小

// array - 固定大小数组（C++11）
std::array<float, 100> fixed_buffer;
fixed_buffer.fill(0.0f);          // 填充所有元素

// deque - 双端队列
std::deque<int> dq;
dq.push_front(1);                 // 头部插入
dq.push_back(2);                  // 尾部插入

// list - 双向链表
std::list<std::string> names;
names.push_back("Alice");
names.push_front("Bob");
```

### 2. 关联容器

```cpp
// map - 有序键值对
std::map<std::string, int> word_freq;
word_freq["hello"] = 1;
word_freq["world"] = 2;

// 遍历map
for (const auto& [word, freq] : word_freq) {
    std::cout << word << ": " << freq << std::endl;
}

// unordered_map - 哈希表（来自sherpa-onnx）
std::unordered_map<std::string, int32_t> token2id;
std::unordered_map<int32_t, std::string> id2token;

// set - 有序集合
std::set<int> unique_numbers;
unique_numbers.insert(5);
unique_numbers.insert(3);
unique_numbers.insert(5);  // 不会重复插入

// multimap/multiset - 允许重复
std::multimap<std::string, std::string> phone_book;
phone_book.emplace("John", "123-456");
phone_book.emplace("John", "789-012");
```

### 3. 容器适配器

```cpp
// stack - 栈
std::stack<int> s;
s.push(10);
s.push(20);
int top = s.top();  // 20
s.pop();

// queue - 队列（来自sherpa-onnx）
std::queue<SpeechSegment> segments;
segments.push(segment);
while (!segments.empty()) {
    auto seg = segments.front();
    segments.pop();
    // 处理段...
}

// priority_queue - 优先队列
std::priority_queue<int> pq;
pq.push(3);
pq.push(1);
pq.push(4);
// 最大元素在顶部

// 自定义比较器
auto cmp = [](const auto& a, const auto& b) {
    return a.score < b.score;
};
std::priority_queue<Result, std::vector<Result>, decltype(cmp)> results(cmp);
```

### 4. 迭代器

```cpp
// 迭代器类型
std::vector<int> vec = {1, 2, 3, 4, 5};

// 正向迭代器
for (auto it = vec.begin(); it != vec.end(); ++it) {
    *it *= 2;
}

// 反向迭代器
for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit) {
    std::cout << *rit << " ";
}

// const迭代器
void PrintVector(const std::vector<int>& v) {
    for (auto cit = v.cbegin(); cit != v.cend(); ++cit) {
        std::cout << *cit << " ";
    }
}

// 迭代器操作
auto it = vec.begin();
std::advance(it, 3);      // 移动迭代器
auto dist = std::distance(vec.begin(), it);  // 计算距离
```

### 5. 算法

```cpp
// 常用算法
std::vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6};

// 排序
std::sort(nums.begin(), nums.end());
std::sort(nums.begin(), nums.end(), std::greater<int>());

// 查找
auto it = std::find(nums.begin(), nums.end(), 5);
if (it != nums.end()) {
    std::cout << "Found at position: " << std::distance(nums.begin(), it);
}

// 变换
std::vector<int> squared;
std::transform(nums.begin(), nums.end(), std::back_inserter(squared),
               [](int n) { return n * n; });

// 累积
int sum = std::accumulate(nums.begin(), nums.end(), 0);

// 条件操作
bool all_positive = std::all_of(nums.begin(), nums.end(),
                               [](int n) { return n > 0; });

// 移除元素
nums.erase(std::remove_if(nums.begin(), nums.end(),
                         [](int n) { return n % 2 == 0; }),
           nums.end());

// 来自sherpa-onnx的例子
std::vector<std::pair<float, int>> score_indices;
// ... 填充score_indices
std::sort(score_indices.begin(), score_indices.end(),
          std::greater<std::pair<float, int>>());
```

## 智能指针

### 1. unique_ptr

```cpp
// 基本使用
std::unique_ptr<int> ptr1(new int(42));
std::unique_ptr<int> ptr2 = std::make_unique<int>(42);  // 推荐方式

// 自定义删除器
auto deleter = [](FILE* f) { if (f) fclose(f); };
std::unique_ptr<FILE, decltype(deleter)> file(fopen("test.txt", "r"), deleter);

// 数组版本
std::unique_ptr<float[]> array = std::make_unique<float[]>(100);

// 来自sherpa-onnx的例子
class VoiceActivityDetector {
private:
    std::unique_ptr<Impl> impl_;
public:
    VoiceActivityDetector(const VadModelConfig &config)
        : impl_(std::make_unique<Impl>(config)) {}
};

// 移动语义
std::unique_ptr<Model> CreateModel() {
    auto model = std::make_unique<Model>();
    // 配置model...
    return model;  // 移动返回
}
```

### 2. shared_ptr

```cpp
// 基本使用
std::shared_ptr<Resource> res1 = std::make_shared<Resource>();
std::shared_ptr<Resource> res2 = res1;  // 共享所有权

// 引用计数
std::cout << "Reference count: " << res1.use_count() << std::endl;

// 自定义删除器
std::shared_ptr<int> array(new int[10], 
                          [](int* p) { delete[] p; });

// 循环引用问题
class Node {
public:
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> parent;  // 使用weak_ptr避免循环引用
};

// 来自sherpa-onnx的例子
using ContextGraphPtr = std::shared_ptr<ContextGraph>;

class OnlineStream {
private:
    ContextGraphPtr context_graph_;
public:
    OnlineStream(ContextGraphPtr graph) 
        : context_graph_(std::move(graph)) {}
};
```

### 3. weak_ptr

```cpp
// weak_ptr基本使用
std::shared_ptr<int> sp = std::make_shared<int>(42);
std::weak_ptr<int> wp = sp;

// 检查是否过期
if (!wp.expired()) {
    // 转换为shared_ptr使用
    if (auto locked = wp.lock()) {
        std::cout << *locked << std::endl;
    }
}

// 观察者模式
class Subject {
private:
    std::vector<std::weak_ptr<Observer>> observers_;
    
public:
    void Attach(std::shared_ptr<Observer> observer) {
        observers_.push_back(observer);
    }
    
    void Notify() {
        auto it = observers_.begin();
        while (it != observers_.end()) {
            if (auto observer = it->lock()) {
                observer->Update();
                ++it;
            } else {
                // 移除已失效的观察者
                it = observers_.erase(it);
            }
        }
    }
};
```

### 4. 智能指针最佳实践

```cpp
// 1. 优先使用make_unique/make_shared
auto ptr = std::make_unique<Widget>();  // 异常安全
// 而不是
// std::unique_ptr<Widget> ptr(new Widget());

// 2. 使用unique_ptr表示独占所有权
class ResourceManager {
private:
    std::unique_ptr<Resource> resource_;
public:
    void SetResource(std::unique_ptr<Resource> res) {
        resource_ = std::move(res);
    }
};

// 3. 使用shared_ptr表示共享所有权
class Cache {
private:
    std::unordered_map<std::string, std::shared_ptr<Data>> cache_;
public:
    std::shared_ptr<Data> Get(const std::string& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        return nullptr;
    }
};

// 4. 避免从原始指针创建多个shared_ptr
int* raw_ptr = new int(42);
// std::shared_ptr<int> sp1(raw_ptr);  // 错误！
// std::shared_ptr<int> sp2(raw_ptr);  // 双重删除！

// 5. enable_shared_from_this
class Widget : public std::enable_shared_from_this<Widget> {
public:
    std::shared_ptr<Widget> GetPtr() {
        return shared_from_this();
    }
};
```

## 异常处理

### 1. 基本异常处理

```cpp
// 抛出异常
void ProcessFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    // 处理文件...
}

// 捕获异常
try {
    ProcessFile("data.txt");
} catch (const std::runtime_error& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
} catch (...) {
    std::cerr << "Unknown exception" << std::endl;
}
```

### 2. 自定义异常

```cpp
// 自定义异常类
class AudioException : public std::exception {
private:
    std::string message_;
    
public:
    explicit AudioException(const std::string& msg) 
        : message_("Audio error: " + msg) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }
};

// 派生更具体的异常
class SampleRateException : public AudioException {
public:
    SampleRateException(int expected, int actual)
        : AudioException("Sample rate mismatch: expected " + 
                        std::to_string(expected) + 
                        ", got " + std::to_string(actual)) {}
};
```

### 3. 异常安全性

```cpp
// 强异常安全保证
class SafeVector {
private:
    std::vector<int> data_;
    
public:
    void Push(int value) {
        // 创建副本，如果分配失败，原数据不变
        std::vector<int> temp = data_;
        temp.push_back(value);
        data_.swap(temp);  // noexcept操作
    }
};

// RAII和异常安全
void ProcessAudio() {
    std::unique_ptr<float[]> buffer(new float[1024]);
    
    // 即使抛出异常，buffer也会被正确释放
    if (!Initialize()) {
        throw std::runtime_error("Initialization failed");
    }
    
    // 使用buffer...
}  // 自动清理

// noexcept说明符
class AudioBuffer {
public:
    // 移动构造函数应该是noexcept的
    AudioBuffer(AudioBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}
    
    // 查询函数通常是noexcept的
    size_t Size() const noexcept { return size_; }
    
private:
    float* data_;
    size_t size_;
};
```

### 4. 错误处理策略

```cpp
// 1. 使用可选类型而非异常
std::optional<int> ParseInt(const std::string& str) {
    try {
        return std::stoi(str);
    } catch (...) {
        return std::nullopt;
    }
}

// 2. 错误码方式（来自sherpa-onnx）
enum class ErrorCode {
    kOk = 0,
    kInvalidConfig,
    kModelLoadFailed,
    kOutOfMemory
};

struct Result {
    ErrorCode error;
    std::string message;
    
    bool IsOk() const { return error == ErrorCode::kOk; }
};

// 3. 断言用于调试
void ProcessSamples(const float* samples, int32_t n) {
    assert(samples != nullptr);
    assert(n > 0);
    // 处理...
}
```

## 并发编程

### 1. 线程基础

```cpp
// 创建线程
#include <thread>
#include <chrono>

void WorkerFunction(int id) {
    std::cout << "Worker " << id << " started\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Worker " << id << " finished\n";
}

// 使用线程
std::thread t1(WorkerFunction, 1);
std::thread t2(WorkerFunction, 2);

t1.join();  // 等待线程完成
t2.join();

// Lambda表达式创建线程
std::thread t3([](){ 
    std::cout << "Lambda thread\n"; 
});
t3.join();
```

### 2. 互斥量和锁

```cpp
// 基本互斥量
std::mutex mtx;
int shared_counter = 0;

void IncrementCounter() {
    std::lock_guard<std::mutex> lock(mtx);
    ++shared_counter;
}  // 自动解锁

// unique_lock提供更多灵活性
void FlexibleLocking() {
    std::unique_lock<std::mutex> lock(mtx);
    // 可以手动解锁和重新锁定
    lock.unlock();
    // 做一些不需要锁的工作
    lock.lock();
}

// 来自sherpa-onnx的例子
class OnlineStream::Impl {
private:
    mutable std::mutex mutex_;
    
public:
    void AcceptWaveform(int32_t sampling_rate, 
                       const float *waveform, int32_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        feat_extractor_.AcceptWaveform(sampling_rate, waveform, n);
    }
    
    int32_t NumFramesReady() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return feat_extractor_.NumFramesReady() - start_frame_index_;
    }
};
```

### 3. 条件变量

```cpp
// 生产者-消费者模式
std::queue<int> data_queue;
std::mutex queue_mutex;
std::condition_variable cv;
bool finished = false;

// 生产者
void Producer() {
    for (int i = 0; i < 10; ++i) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            data_queue.push(i);
        }
        cv.notify_one();  // 通知消费者
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        finished = true;
    }
    cv.notify_all();  // 通知所有消费者
}

// 消费者
void Consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv.wait(lock, []{ return !data_queue.empty() || finished; });
        
        while (!data_queue.empty()) {
            int value = data_queue.front();
            data_queue.pop();
            lock.unlock();
            
            // 处理数据
            std::cout << "Consumed: " << value << std::endl;
            
            lock.lock();
        }
        
        if (finished) break;
    }
}
```

### 4. 原子操作

```cpp
// 原子变量
std::atomic<int> atomic_counter{0};
std::atomic<bool> ready{false};

void Worker() {
    while (!ready.load()) {
        std::this_thread::yield();
    }
    
    for (int i = 0; i < 1000; ++i) {
        atomic_counter.fetch_add(1);  // 原子递增
    }
}

// 内存序
std::atomic<int> data;
std::atomic<bool> flag{false};

// 写线程
void Writer() {
    data.store(42, std::memory_order_relaxed);
    flag.store(true, std::memory_order_release);
}

// 读线程
void Reader() {
    while (!flag.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    int value = data.load(std::memory_order_relaxed);
}
```

### 5. 异步编程

```cpp
// future和promise
std::promise<int> promise;
std::future<int> future = promise.get_future();

// 在另一个线程中设置值
std::thread t([&promise]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    promise.set_value(42);
});

// 获取结果
int result = future.get();  // 阻塞直到值可用
t.join();

// async
auto future_result = std::async(std::launch::async, []() {
    // 在另一个线程中执行
    return ComputeExpensiveValue();
});

// 做其他工作...

// 获取结果
auto value = future_result.get();

// packaged_task
std::packaged_task<int(int, int)> task([](int a, int b) {
    return a + b;
});

std::future<int> task_future = task.get_future();
std::thread task_thread(std::move(task), 10, 20);

int sum = task_future.get();  // 30
task_thread.join();
```

---

## 总结

第三部分介绍了C++的高级特性：

1. **模板编程**：函数模板、类模板、模板元编程、可变参数模板
2. **STL容器**：序列容器、关联容器、容器适配器、迭代器、算法
3. **智能指针**：unique_ptr、shared_ptr、weak_ptr、最佳实践
4. **异常处理**：基本异常处理、自定义异常、异常安全性、错误处理策略
5. **并发编程**：线程、互斥量、条件变量、原子操作、异步编程

这些高级特性使C++成为一个功能强大的编程语言，能够处理各种复杂的编程任务。通过学习sherpa-onnx项目中的实际代码，我们可以看到这些特性在实际项目中的应用。

## 进一步学习建议

1. **实践项目**：尝试使用这些特性编写自己的音频处理程序
2. **阅读源码**：深入研究sherpa-onnx的实现细节
3. **性能优化**：学习如何使用这些特性优化代码性能
4. **现代C++**：关注C++17/20/23的新特性
5. **设计模式**：学习如何在C++中实现常见的设计模式

掌握这些高级特性将使您能够编写高效、安全、可维护的C++代码，充分发挥C++语言的强大能力。