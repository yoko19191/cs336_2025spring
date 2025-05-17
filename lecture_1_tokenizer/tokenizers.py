"""
实现各种分词器：
1. CharacterTokenizer - 字符级分词器
2. ByteTokenizer - 字节级分词器
3. WordTokenizer - 单词级分词器
4. BPETokenizer - 字节对编码分词器
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set

GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BaseTokenizer:
    """基础分词器类"""
    
    def tokenize(self, text: str) -> List[str]:
        """将文本转换为token列表"""
        raise NotImplementedError
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID列表"""
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """将token ID列表解码为文本"""
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids]
        return "".join(tokens)


class CharacterTokenizer(BaseTokenizer):
    """字符级分词器"""
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.unk_id = 0
        
        # 将未知字符映射到ID 0
        self.token_to_id["<UNK>"] = self.unk_id
        self.id_to_token[self.unk_id] = "<UNK>"
        
        # 构建基本字符表（ASCII和常见Unicode）
        for i, char in enumerate(range(32, 127), start=1):  # ASCII可打印字符
            self.token_to_id[chr(char)] = i
            self.id_to_token[i] = chr(char)
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分解为字符列表"""
        return list(text)


class ByteTokenizer(BaseTokenizer):
    """UTF-8字节级分词器"""
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.unk_id = 0
        
        # 将未知字节映射到ID 0
        self.token_to_id["<UNK>"] = self.unk_id
        self.id_to_token[self.unk_id] = "<UNK>"
        
        # 构建字节表（0-255）
        for i in range(256):
            self.token_to_id[i] = i + 1  # 从1开始编号
            self.id_to_token[i + 1] = i
    
    def tokenize(self, text: str) -> List[int]:
        """将文本分解为UTF-8字节列表"""
        # 直接返回UTF-8字节值
        return list(text.encode('utf-8'))
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID列表"""
        bytes_data = text.encode('utf-8')
        return [self.token_to_id.get(b, self.unk_id) for b in bytes_data]
    
    def decode(self, ids: List[int]) -> str:
        """将token ID列表解码为文本"""
        bytes_data = bytes([self.id_to_token.get(id, 0) for id in ids])
        try:
            return bytes_data.decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            return bytes_data.decode('utf-8', errors='replace')


class WordTokenizer(BaseTokenizer):
    """单词级分词器"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.unk_id = 0
        self.word_pattern = re.compile(r'\b\w+\b|[^\w\s]')
        
        # 特殊标记
        self.token_to_id["<UNK>"] = self.unk_id
        self.id_to_token[self.unk_id] = "<UNK>"
    
    def build_vocab(self, texts: List[str]):
        """从文本集合中构建词汇表"""
        # 统计词频
        counter = Counter()
        for text in texts:
            words = self.word_pattern.findall(text.lower())
            counter.update(words)
        
        # 选择最常见的词构建词汇表
        for i, (word, _) in enumerate(counter.most_common(self.vocab_size - 1), start=1):
            self.token_to_id[word] = i
            self.id_to_token[i] = word
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分解为单词列表"""
        return self.word_pattern.findall(text.lower())
    
    def decode(self, ids: List[int]) -> str:
        """将token ID列表解码为文本"""
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids]
        return " ".join(tokens)


class BPETokenizer(BaseTokenizer):
    """字节对编码(BPE)分词器"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.unk_id = 0
        self.merges = {}  # 存储合并规则
        
        # 特殊标记
        self.token_to_id["<UNK>"] = self.unk_id
        self.id_to_token[self.unk_id] = "<UNK>"
    
    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """计算相邻符号对的频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """根据给定的pair合并词汇表"""
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_vocab = {}
        
        for word, freq in vocab.items():
            parts = word.split()
            i = 0
            new_parts = []
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                    new_parts.append(replacement)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            new_word = ' '.join(new_parts)
            new_vocab[new_word] = freq
        
        return new_vocab
    
    def train(self, texts: List[str], num_merges: int = 10000):
        """训练BPE模型"""
        # 初始词汇表：将每个单词分解为字符
        vocab = Counter()
        for text in texts:
            for word in text.split():
                # 将单词表示为用空格分隔的字符序列
                chars = ' '.join(list(word))
                vocab[chars] += 1
        
        # 初始词汇表包含所有单个字符
        vocab_size = len(set(''.join(vocab.keys())))
        
        # 构建初始字符词汇表
        for i, char in enumerate(sorted(set(''.join(vocab.keys())))):
            self.token_to_id[char] = i + 1  # ID从1开始
            self.id_to_token[i + 1] = char
        
        # 开始进行合并操作
        merges_count = 0
        for i in range(min(num_merges, self.vocab_size - vocab_size)):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
                
            # 找出最常见的对
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            
            # 记录合并规则
            self.merges[best_pair] = ''.join(best_pair)
            
            # 将新token添加到词汇表
            new_token = ''.join(best_pair)
            if new_token not in self.token_to_id:
                token_id = len(self.token_to_id) + 1
                self.token_to_id[new_token] = token_id
                self.id_to_token[token_id] = new_token
                
            merges_count += 1
            if len(self.token_to_id) >= self.vocab_size:
                break
    
    def tokenize(self, text: str) -> List[str]:
        """使用BPE对文本进行分词"""
        tokens = []
        for word in text.split():
            word_tokens = list(word)  # 初始化为字符列表
            
            # 应用合并规则
            while True:
                # 查找所有可能的合并对
                pairs = {}
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    if pair in self.merges:
                        pairs[i] = pair
                
                if not pairs:
                    break
                    
                # 执行最先出现的合并
                first_idx = min(pairs.keys())
                pair = pairs[first_idx]
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i == first_idx:
                        new_tokens.append(self.merges[pair])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            tokens.extend(word_tokens)
        
        return tokens 