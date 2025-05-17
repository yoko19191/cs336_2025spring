"""
分词器使用示例
"""

from lecture_1_tokenizer.tokenizers import (
    CharacterTokenizer,
    ByteTokenizer,
    WordTokenizer,
    BPETokenizer
)

def print_tokens(tokenizer_name, tokens):
    """以友好的形式打印分词结果"""
    print(f"{tokenizer_name} 分词结果:")
    print(f"  Token数量: {len(tokens)}")
    
    # 对于字节分词器，显示十六进制值
    if tokenizer_name == "字节级分词器 (UTF-8)":
        hex_tokens = [f"0x{t:02x}" if isinstance(t, int) else t for t in tokens[:20]]
        print(f"  Tokens (HEX): {hex_tokens}..." if len(tokens) > 20 else f"  Tokens (HEX): {hex_tokens}")
    else:
        print(f"  Tokens: {tokens[:20]}..." if len(tokens) > 20 else f"  Tokens: {tokens}")
    print()

def main():
    """测试各种分词器"""
    sample_text = "自然语言处理(NLP)是人工智能领域的重要分支。分词是NLP的基础任务之一。"
    
    print(f"原始文本: {sample_text}")
    print(f"文本长度: {len(sample_text)} 个字符")
    print()
    
    # 测试字符级分词器
    char_tokenizer = CharacterTokenizer()
    char_tokens = char_tokenizer.tokenize(sample_text)
    print_tokens("字符级分词器", char_tokens)
    
    # 测试字节级分词器
    byte_tokenizer = ByteTokenizer()
    byte_tokens = byte_tokenizer.tokenize(sample_text)
    print_tokens("字节级分词器 (UTF-8)", byte_tokens)
    
    # 演示UTF-8编码中文字符需要多个字节
    chinese_char = "自"
    char_tokens = CharacterTokenizer().tokenize(chinese_char)
    byte_tokens = ByteTokenizer().tokenize(chinese_char)
    
    print(f"中文字符 '{chinese_char}' 的编码:")
    print(f"  字符分词器: {char_tokens} (1个token)")
    print(f"  字节分词器: {[f'0x{b:02x}' for b in byte_tokens]} ({len(byte_tokens)}个token)")
    print()
    
    # 测试单词级分词器（简单模式，无需训练）
    word_tokenizer = WordTokenizer()
    word_tokens = word_tokenizer.tokenize(sample_text)
    print_tokens("单词级分词器", word_tokens)
    
    # 测试BPE分词器（需要训练）
    # 准备一些训练文本
    train_texts = [
        "自然语言处理是计算机科学的一个重要领域",
        "分词是自然语言处理的基础任务之一",
        "机器学习和深度学习在自然语言处理中发挥重要作用",
        "Transformer模型已成为自然语言处理的主流模型",
        "BERT, GPT, T5等都是基于Transformer的模型"
    ]
    
    bpe_tokenizer = BPETokenizer(vocab_size=100)
    bpe_tokenizer.train(train_texts, num_merges=50)
    bpe_tokens = bpe_tokenizer.tokenize(sample_text)
    print_tokens("BPE分词器", bpe_tokens)
    
    # 展示编码和解码
    print("编码和解码示例:")
    text_to_encode = "自然语言"
    
    # 字符级
    char_ids = char_tokenizer.encode(text_to_encode)
    char_decoded = char_tokenizer.decode(char_ids)
    print(f"  字符级: '{text_to_encode}' -> {char_ids} -> '{char_decoded}'")
    
    # 字节级
    byte_ids = byte_tokenizer.encode(text_to_encode)
    byte_decoded = byte_tokenizer.decode(byte_ids)
    print(f"  字节级: '{text_to_encode}' -> {byte_ids} -> '{byte_decoded}'")

if __name__ == "__main__":
    main() 