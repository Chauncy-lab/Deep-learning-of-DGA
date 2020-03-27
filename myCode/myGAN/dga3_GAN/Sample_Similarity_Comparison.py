import numpy as np

import myCode.dga3_GAN.language_helpers

#GAN与WGAN生成的样本相似度：1-gram，2-gram，3-gram，4-gram
#GAN best权重产出的最佳样本 与 wgan-gp 3万轮后的样本，分别于正常样本Alexa 求js的散度，发现wgan-gp的js散度
#无论是n = 1，2，3，4 gram的提取，都是wgan-gp与Alexa的样本相近

# 加载原数据
DATA_DIR = 'AlexaTop1M_NoSeparate'

def load_dataset_Alexa(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir='/home/ishaan/data/1-billion-word-language-modeling-benchmark-r13output'):
    print ("loading dataset_Alexa...")

    lines = []

    finished = False

    for i in range(100):
        path = data_dir+("/TopDomainName.Less33-{}0000-OF-996720".format(str(i).zfill(2)))
        # If you using chinese-based OS, encoding default will be Big-5, so You should change it to utf-8
        with open(path, 'r',encoding = 'utf8') as f:
            for line in f:
                line = line[:-1]
                b = line.split('.')
                line = b[0]
                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ("`",)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break


    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in range(100):
        print( filtered_lines[i] )

    print ( "loaded {} lines in dataset".format(len(lines)) )
    return filtered_lines, charmap, inv_charmap


def load_dataset_sample(max_length, max_n_examples, tokenize=False, max_vocab_size=2048):
    print ("loading dataset_sample...")

    lines = []

    finished = False

    for i in range(100):
        path = 'GAN_samples.txt'
        # If you using chinese-based OS, encoding default will be Big-5, so You should change it to utf-8
        with open(path, 'r',encoding = 'utf8') as f:
            for line in f:
                line = line[:-1]
                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ("`",)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break


    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in range(100):
        print( filtered_lines[i] )

    print ( "loaded {} lines in dataset".format(len(lines)) )
    return filtered_lines, charmap, inv_charmap

def tokenize_string(sample):
    a = tuple(sample.lower().split(' '))
    return a


Alexa_lines, Alexa_charmap, Alexa_inv_charmap =load_dataset_Alexa(
    max_length=32,
    max_n_examples=100000,
    data_dir=DATA_DIR
)
sample_lines, sample_charmap, sample_inv_charmap =load_dataset_sample(
    max_length=32,
    max_n_examples=640,
)

# 真实的1-4gram
alexa_char_ngram_lms = [myCode.dga3_GAN.language_helpers.NgramLanguageModel(i+1, Alexa_lines, tokenize=False) for i in range(4)]
# GAN生产出来的 1-4gram
sample_char_ngram_lms = [myCode.dga3_GAN.language_helpers.NgramLanguageModel(i+1, sample_lines, tokenize=False) for i in range(4)]


for i in range(4):
    a = sample_char_ngram_lms[i].js_with(alexa_char_ngram_lms[i])
    print(str(i) + ":"+ str(a))