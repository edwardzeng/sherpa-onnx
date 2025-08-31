# C++学习指南 - 基于sherpa-onnx项目

本文档整理了sherpa-onnx项目中的常见C++语法模式和编程技巧，帮助学习者通过实际项目代码理解C++的使用。

## 目录

1. [项目概述](#项目概述)
2. [命名空间和头文件组织](#命名空间和头文件组织)
3. [类设计模式](#类设计模式)
4. [智能指针使用](#智能指针使用)
5. [模板编程](#模板编程)
6. [错误处理和日志](#错误处理和日志)
7. [音频处理相关](#音频处理相关)
8. [字符串处理](#字符串处理)
9. [宏定义技巧](#宏定义技巧)
10. [API设计模式](#api设计模式)

## 项目概述

sherpa-onnx是一个高性能的语音处理框架，支持：
- 语音识别（ASR）
- 语音合成（TTS）
- 说话人识别
- 语音活动检测（VAD）
- 关键词检测（KWS）

项目使用现代C++特性，展示了工业级C++代码的组织方式。

## 命名空间和头文件组织

### 1. 头文件保护

```cpp
#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_
// ... 头文件内容
#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_
```

### 2. 命名空间使用

```cpp
namespace sherpa_onnx {
  // 主命名空间
}

namespace sherpa_onnx::cxx {
  // C++ API的子命名空间
}

// 在实现文件中使用
using namespace sherpa_onnx::cxx;  // NOLINT
```

### 3. 标准头文件包含

```cpp
#include <chrono>    // 时间处理
#include <iostream>  // 输入输出
#include <memory>    // 智能指针
#include <string>    // 字符串
#include <vector>    // 动态数组
```

## 类设计模式

### 1. 接口-实现分离（Pimpl习语）

```cpp
// 头文件中
class VoiceActivityDetector {
 public:
  explicit VoiceActivityDetector(const VadModelConfig &config,
                                 float buffer_size_in_seconds = 30);
  ~VoiceActivityDetector();
  
  // 公共接口
  void AcceptWaveform(const float *samples, int32_t n) const;
  bool IsEmpty() const;
  
 private:
  // 隐藏实现细节
  std::unique_ptr<Impl> impl_;
};
```

### 2. 工厂模式

```cpp
// 静态工厂方法
class OnlineRecognizer : public MoveOnly<OnlineRecognizer, SherpaOnnxOnlineRecognizer> {
 public:
  static OnlineRecognizer Create(const OnlineRecognizerConfig &config) {
    SherpaOnnxOnlineRecognizerConfig c = ToConfig(config);
    SherpaOnnxOnlineRecognizer *p = SherpaOnnxCreateOnlineRecognizer(&c);
    return OnlineRecognizer(p);
  }
};
```

### 3. 配置类设计

```cpp
struct OnlineRecognizerConfig {
  FeatureConfig feat_config;
  OnlineModelConfig model_config;
  
  std::string decoding_method = "greedy_search";  // 默认值
  int32_t max_active_paths = 4;
  bool enable_endpoint = false;
  
  // 配置验证
  void Validate() const;
  
  // 序列化
  std::string ToString() const;
};
```

## 智能指针使用

### 1. unique_ptr用于独占所有权

```cpp
// 创建unique_ptr
std::unique_ptr<OnlineStream> CreateStream() const override {
  return std::make_unique<OnlineStream>(feat_config);
}

// 在类中存储
class SpeakerEmbeddingExtractor {
 private:
  std::unique_ptr<Impl> impl_;
};
```

### 2. shared_ptr用于共享所有权

```cpp
// 使用别名简化类型
using ContextGraphPtr = std::shared_ptr<ContextGraph>;

// 创建和传递
ContextGraphPtr context_graph = std::make_shared<ContextGraph>(keywords);
```

### 3. RAII和移动语义

```cpp
template <typename Derived, typename T>
class MoveOnly {
 public:
  MoveOnly() = default;
  explicit MoveOnly(const T *p) : p_(p) {}
  
  ~MoveOnly() { Destroy(); }
  
  // 禁用拷贝
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly &operator=(const MoveOnly &) = delete;
  
  // 启用移动
  MoveOnly(MoveOnly &&other) : p_(other.Release()) {}
  MoveOnly &operator=(MoveOnly &&other) {
    if (&other == this) return *this;
    Destroy();
    p_ = other.Release();
    return *this;
  }
  
 private:
  const T *p_ = nullptr;
};
```

## 模板编程

### 1. 函数模板

```cpp
// 字符串转整数的通用模板
template <class Int>
bool ConvertStringToInteger(const std::string &str, Int *out) {
  static_assert(std::is_integral<Int>::value, "");
  // 实现...
}

// 分割字符串为整数向量
template <class I>
bool SplitStringToIntegers(const std::string &full, const char *delim,
                          bool omit_empty_strings, std::vector<I> *out) {
  static_assert(std::is_integral<I>::value, "");
  // 实现...
}
```

### 2. 模板特化和SFINAE

```cpp
// 管理器模板参数
template <typename Manager>
explicit VoiceActivityDetector(Manager *mgr, const VadModelConfig &config,
                              float buffer_size_in_seconds = 30);

// 显式实例化
template VoiceActivityDetector::VoiceActivityDetector(
    ModelManager *mgr, const VadModelConfig &config,
    float buffer_size_in_seconds);
```

## 错误处理和日志

### 1. 条件编译的日志宏

```cpp
#if __ANDROID_API__ >= 8
  #define SHERPA_ONNX_LOGE(...)                                            \
    do {                                                                   \
      fprintf(stderr, "%s:%s:%d ", __FILE__, __func__, __LINE__);         \
      fprintf(stderr, ##__VA_ARGS__);                                      \
      fprintf(stderr, "\n");                                               \
      __android_log_print(ANDROID_LOG_WARN, "sherpa-onnx", ##__VA_ARGS__); \
    } while (0)
#else
  #define SHERPA_ONNX_LOGE(...) /* 其他平台实现 */
#endif
```

### 2. 错误检查模式

```cpp
// 创建和检查
OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
if (!recognizer.Get()) {
  std::cerr << "Please check your config\n";
  return -1;
}

// 文件读取检查
Wave wave = ReadWave(wave_filename);
if (wave.samples.empty()) {
  std::cerr << "Failed to read: '" << wave_filename << "'\n";
  return -1;
}
```

### 3. 异常安全的资源管理

```cpp
// 使用RAII确保资源释放
class FileHandle {
 public:
  explicit FileHandle(const std::string &filename) 
    : file_(fopen(filename.c_str(), "rb")) {
    if (!file_) {
      SHERPA_ONNX_LOGE("Failed to open file: %s", filename.c_str());
    }
  }
  
  ~FileHandle() {
    if (file_) fclose(file_);
  }
  
  // 禁用拷贝，启用移动
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(const FileHandle&) = delete;
  FileHandle(FileHandle&&) = default;
  FileHandle& operator=(FileHandle&&) = default;
  
 private:
  FILE* file_;
};
```

## 音频处理相关

### 1. 波形数据结构

```cpp
struct Wave {
  std::vector<float> samples;  // 音频样本
  int32_t sample_rate;         // 采样率
};

// 读写音频文件
Wave ReadWave(const std::string &filename);
bool WriteWave(const std::string &filename, const Wave &wave);
```

### 2. 流式处理接口

```cpp
class OnlineStream {
 public:
  // 接收音频数据
  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const;
  
  // 标记输入结束
  void InputFinished() const;
  
  // 获取特征帧
  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const;
  
  // 重置流状态
  void Reset();
};
```

### 3. 特征提取配置

```cpp
struct FeatureExtractorConfig {
  int32_t sampling_rate = 16000;
  int32_t feature_dim = 80;
  float low_freq = 20.0f;
  float high_freq = -400.0f;
  
  bool normalize_samples = true;
  float frame_shift_ms = 10.0f;
  float frame_length_ms = 25.0f;
  
  // 注册命令行参数
  void Register(ParseOptions *po);
  
  // 验证配置
  void Validate() const;
};
```

## 字符串处理

### 1. 字符串分割

```cpp
// 基本字符串分割
void SplitStringToVector(const std::string &full, const char *delim,
                        bool omit_empty_strings,
                        std::vector<std::string> *out);

// 使用示例
std::vector<std::string> tokens;
SplitStringToVector("hello,world,test", ",", false, &tokens);
// tokens = ["hello", "world", "test"]
```

### 2. 字符串转换

```cpp
// 安全的字符串转整数
template <class Int>
bool ConvertStringToInteger(const std::string &str, Int *out) {
  const char *this_str = str.c_str();
  char *end = nullptr;
  errno = 0;
  int64_t i = strtoll(this_str, &end, 10);
  
  // 检查转换是否成功
  if (end == this_str || *end != '\0' || errno != 0) 
    return false;
    
  // 检查范围
  Int iInt = static_cast<Int>(i);
  if (static_cast<int64_t>(iInt) != i) 
    return false;
    
  *out = iInt;
  return true;
}
```

### 3. UTF-8处理

```cpp
// 项目使用utfcpp库处理UTF-8
#include "utf8.h"

// 验证UTF-8字符串
bool is_valid = utf8::is_valid(text.begin(), text.end());

// 计算UTF-8字符数
size_t char_count = utf8::distance(text.begin(), text.end());
```

## 宏定义技巧

### 1. 元数据读取宏

```cpp
// 读取整数元数据
#define SHERPA_ONNX_READ_META_DATA(dst, src_key)                           \
  do {                                                                     \
    auto value = LookupCustomModelMetaData(meta_data, src_key, allocator); \
    if (value.empty()) {                                                   \
      SHERPA_ONNX_LOGE("'%s' does not exist in the metadata", src_key);    \
      SHERPA_ONNX_EXIT(-1);                                                \
    }                                                                      \
    dst = atoi(value.c_str());                                             \
  } while (0)

// 使用示例
int32_t vocab_size;
SHERPA_ONNX_READ_META_DATA(vocab_size, "vocab_size");
```

### 2. 平台相关的宏

```cpp
// 跨平台的字符串转长整型
#ifdef _MSC_VER
  #define SHERPA_ONNX_STRTOLL(cur_cstr, end_cstr) \
    _strtoi64(cur_cstr, end_cstr, 10);
#else
  #define SHERPA_ONNX_STRTOLL(cur_cstr, end_cstr) \
    strtoll(cur_cstr, end_cstr, 10);
#endif
```

## API设计模式

### 1. 流畅接口（Fluent Interface）

```cpp
// 链式调用配置
OnlineRecognizerConfig config;
config.feat_config.sampling_rate = 16000;
config.model_config.num_threads = 4;
config.enable_endpoint = true;

// 创建识别器
auto recognizer = OnlineRecognizer::Create(config);
```

### 2. 结果封装

```cpp
struct OnlineRecognizerResult {
  std::string text;                    // 识别文本
  std::vector<std::string> tokens;     // 词元列表
  std::vector<float> timestamps;       // 时间戳
  
  // JSON序列化
  std::string AsJsonString() const {
    // 返回JSON格式的结果
  }
};
```

### 3. 回调和观察者模式

```cpp
// VAD示例：处理检测到的语音段
while (!vad.IsEmpty()) {
  auto segment = vad.Front();
  float start_time = segment.start / static_cast<float>(sample_rate);
  float end_time = start_time + 
    segment.samples.size() / static_cast<float>(sample_rate);
  
  printf("%.3f -- %.3f\n", start_time, end_time);
  
  // 处理语音段...
  
  vad.Pop();
}
```

### 4. 批处理优化

```cpp
// 并行处理多个流
std::vector<OnlineStream*> streams;
// ... 创建多个流

// 批量解码
while (recognizer.IsReady(streams)) {
  recognizer.DecodeStreams(streams.data(), streams.size());
}

// 获取结果
for (auto* stream : streams) {
  auto result = recognizer.GetResult(stream);
  // 处理结果...
}
```

## 最佳实践总结

### 1. 资源管理
- 使用RAII管理资源
- 优先使用智能指针而非裸指针
- 实现移动语义，禁用不必要的拷贝

### 2. 错误处理
- 使用返回值或可选类型表示错误
- 提供详细的错误日志
- 在关键位置进行参数验证

### 3. 性能优化
- 使用const引用传递大对象
- 实现移动语义避免不必要的拷贝
- 合理使用内联函数

### 4. 代码组织
- 接口与实现分离
- 使用命名空间避免名称冲突
- 遵循一致的命名规范

### 5. 跨平台考虑
- 使用条件编译处理平台差异
- 抽象平台相关功能
- 提供统一的API接口

## 学习建议

1. **从简单示例开始**：先看`cxx-api-examples`目录下的示例代码
2. **理解核心概念**：重点学习`OnlineRecognizer`、`OnlineStream`等核心类
3. **关注设计模式**：学习项目中的工厂模式、Pimpl等设计模式
4. **实践音频处理**：尝试修改音频处理相关的代码
5. **阅读测试代码**：通过测试代码理解API的使用方式

## 扩展阅读

- [sherpa-onnx官方文档](https://k2-fsa.github.io/sherpa/)
- [ONNX Runtime文档](https://onnxruntime.ai/docs/)
- [现代C++教程](https://changkun.de/modern-cpp/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)