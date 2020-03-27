from __future__ import division
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
import os


def gen_char_histogram(filepath):
    exp_dir,filename = os.path.split(filepath)
    _,exp_no = os.path.split(exp_dir)

    df = pd.read_csv(filepath, header=None)[0].sample(n=10000).values.tolist()
    cnt = Counter()
    for words in df:
        for letters in set(words):
            cnt[letters] += 1

    d = sum(cnt.values())
    # print(cnt)

    for key, value in cnt.iteritems():
        cnt[key] = value / d

    df = pd.DataFrame.from_dict(cnt, orient='index').sort_index()
    print(df)
    df.plot(kind='bar', title="Characters distribution for %s" % exp_no,sort_columns=True, rot=0)
    plt.show()


if __name__ == '__main__':
    gen_char_histogram("/home/archeffect/PycharmProjects/adversarial_DGA/resources/datasets/all_legit.txt")
