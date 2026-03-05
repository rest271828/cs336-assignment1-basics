from collections import Counter, defaultdict
from typing import Dict, List, Tuple

Token = bytes
TokenSeq = Tuple[Token, ...]
Pair = Tuple[Token, Token]

def word_to_byte_token(word:str)->TokenSeq:
    bs = word.encode('utf-8')
    return tuple(bytes([b]) for b in bs)

def build_dict(corpus:List[str])->Dict[TokenSeq, int]:
    """
    输入: ["low lower", "newer newest"]
    输出: { (b'l', b'o', b'w'): 1, (b'l', b'o', b'w', b'e', b'r'): 1, ... }
    """
    freq_words = Counter()
    for line in corpus:
        for w in line.strip().split():
            if w:
                freq_words[w] += 1

    token_dict: Dict[TokenSeq, int] = {}
    for w, f in freq_words.items():
        seq = word_to_byte_token(w)
        token_dict[seq] = token_dict.get(seq, 0) + f    #f “这个词在语料中出现的总次数”
    print(token_dict)
    return token_dict

def init_vocab() -> Dict[Token, int]:
    '''
    初始词表: 单字节0-255
    '''
    return {bytes([i]): i for i in range(256)}

def pair_stats(token_dict: Dict[TokenSeq, int]) -> Counter:
    '''
    统计所有相邻pair
    返回counter({(b'l',b'o',):7...})
    '''
    stats = Counter()
    for seq, freq in token_dict.items():
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            stats[(seq[i], seq[i + 1])] += freq
    return stats

def argmax(stats:Counter) -> Pair:
    '''
    stats.most_common(1)[0][0] 就是出现最多的pair
    '''
    return stats.most_common(1)[0][0]

def merge_seq(seq: TokenSeq, best: Pair) -> TokenSeq:
    '''
    在一个序列里面把best = (a,b) 合并成a+b
    '''
    a ,b = best 
    merged = a + b

    out: List[Token] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best:
            out.append(merged)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)

def merge_pair(token_dict: Dict[TokenSeq, int], best: Pair) -> Dict[TokenSeq, int]:
    '''
    对整个 token_dict 进行 merge
    '''
    new_dict: Dict[TokenSeq, int] = {}
    for seq, freq in token_dict.items():
        new_seq = merge_seq(seq, best)
        new_dict[new_seq] = new_dict.get(new_seq, 0) + freq
    return new_dict

def update_vocab(vocab: Dict[Token, int], best: Pair) -> Dict[Token, int]:
    '''
    把新的token (a + b) 加入vocab, 分配新的id
    '''
    a, b = best
    merged = a + b
    if merged not in vocab:
        vocab[merged] = max(vocab.values()) +1
    return vocab

def train(corpus, num_merges):
    '''
    step1: 字节化与初始化
    step2: 构建统计字典
    step3: 最频繁字节对
    step4: 合并与记录
    step5: 迭代
    '''
    token_dict = build_dict(corpus)
    merges = []
    vocab = init_vocab()

    # for _ in range(num_merges):
    #     stats = pair_stats(token_dict)
    #     if not stats: break
    #     best = argmax(stats)
    #     merges.append(best)
    #     token_dict = merge_pair(token_dict, best)
    #     vocab = update_vocab(vocab, best)
    for step in range(num_merges):
        stats = pair_stats(token_dict)
        if not stats:
            break
        best = argmax(stats)
        merges.append(best)
        best_count = stats[best]

        print(f"\n=== merge step {step} ===")
        print("best:", show_pair(best), "count:", best_count)

        # 看看合并前有哪些词（挑前几个）
        print("before:")
        for s, f in list(token_dict.items())[:5]:
            print(" ", f, show_seq(s))

        token_dict = merge_pair(token_dict, best)

        print("after:")
        for s, f in list(token_dict.items())[:5]:
            print(" ", f, show_seq(s))

        vocab = update_vocab(vocab, best)

    return merges, vocab

def show_seq(seq: TokenSeq) -> str:
    return "|".join(t.decode("utf-8", errors="replace")for t in seq)


def show_pair(p: Pair) -> str:
    return f"({p[0].decode('utf-8', 'replace')},{p[1].decode('utf-8', 'replace')})"

if __name__ == "__main__":
    corpus = ["low lower", "newer newest"]
    merges, vocab = train(corpus, num_merges=10)
    print("\nCase1", show_seq(merge_seq((b'a', b'b', b'a'), (b'a', b'b'))))
    print("\nCase2", show_seq(merge_seq((b'a', b'b', b'b', b'c'), (b'b', b'b'))))
    print("\nCase3", show_seq(merge_seq((b'a',), (b'a', b'b'))))
    print("\nCase4", show_seq(merge_seq((b'a', b'b', b'a', b'b'), (b'a', b'b'))))
