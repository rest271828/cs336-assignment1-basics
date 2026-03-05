corpus = ["low lower", "newer newest"]

from collections import Counter

freq_words = Counter()

for line in corpus:
    print("原始行:", line)

    cleaned = line.strip()
    print("去空白:", cleaned)

    words = cleaned.split()
    print("切词:", words)

    for w in words:
        if w:
            freq_words[w] += 1

print("最终统计:", freq_words)