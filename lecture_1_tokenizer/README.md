# 分词器实现

这个库实现了四种常见的分词器：

1. 字符级分词器 (CharacterTokenizer)
2. 字节级分词器 (ByteTokenizer)
3. 单词级分词器 (WordTokenizer)
4. 字节对编码分词器 (BPETokenizer)

## 安装

无需特殊安装，直接调用即可。

## 使用示例

### 字符级分词器

```python
from lecture_1_tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer()
tokens = tokenizer.tokenize("自然语言处理")
print(tokens)  # ['自', '然', '语', '言', '处', '理']

# 编码为ID
ids = tokenizer.encode("自然语言处理")
print(ids)  # [可能会包含未知字符的ID]

# 解码回文本
text = tokenizer.decode(ids)
print(text)  # "自然语言处理"
```

### 字节级分词器 (UTF-8)

```python
from lecture_1_tokenizer import ByteTokenizer

tokenizer = ByteTokenizer()
tokens = tokenizer.tokenize("NLP")
print(tokens)  # 将返回UTF-8字节级别的token (整数值)
print([f"0x{b:02x}" for b in tokens])  # 以十六进制显示

# 编码为ID
ids = tokenizer.encode("NLP")
print(ids)

# 解码回文本
text = tokenizer.decode(ids)
print(text)  # "NLP"

# 中文字符用UTF-8编码需要多个字节
tokens = tokenizer.tokenize("自")
print([f"0x{b:02x}" for b in tokens])  # 可能输出 ['0xe8', '0x87', '0xaa']
```

### 单词级分词器

```python
from lecture_1_tokenizer import WordTokenizer

# 创建分词器
tokenizer = WordTokenizer(vocab_size=1000)

# 构建词汇表（从训练文本中）
train_texts = ["自然语言处理是人工智能的重要分支", "分词是NLP的基础任务"]
tokenizer.build_vocab(train_texts)

# 分词
tokens = tokenizer.tokenize("自然语言处理技术正在飞速发展")
print(tokens)

# 编码和解码
ids = tokenizer.encode("自然语言处理")
text = tokenizer.decode(ids)
print(text)
```

### 字节对编码分词器

```python
from lecture_1_tokenizer import BPETokenizer

# 创建分词器
tokenizer = BPETokenizer(vocab_size=1000)

# 训练BPE模型
train_texts = [
    "自然语言处理是计算机科学的分支",
    "机器学习是人工智能的核心",
    "深度学习在NLP领域取得了突破性进展"
]
tokenizer.train(train_texts, num_merges=100)

# 分词
tokens = tokenizer.tokenize("自然语言处理非常重要")
print(tokens)

# 编码和解码
ids = tokenizer.encode("自然语言")
text = tokenizer.decode(ids)
print(text)
```

## 运行示例

```bash
python -m lecture_1_tokenizer.example
```

这将运行示例代码，展示所有四种分词器的效果。 