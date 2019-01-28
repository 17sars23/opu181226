# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging
import sys
import argparse
import random


def make_wiki_model():
    # wikiモデルの作成
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus('./model/wiki_wakati.txt')

    model = word2vec.Word2Vec(sentences, size=200, min_count=20, window=15)
    model.save("./model/wiki.model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--positive',
                        default=['パン', '美味しい', 'まずい'], help='Input positive words.')
    args = parser.parse_args()

    # 学習済みモデルの読み込み
    # 今回は以下のカメリオが公開しているモデルを使用
    # http://aial.shiroyagi.co.jp/2017/02/japanese-word2vec-model-builder/
    model = word2vec.Word2Vec.load("./model/word2vec.gensim.model")

    # wikiモデルの作成と読み込み
    # make_wiki_model()
    # wiki_model = word2vec.Word2Vec.load("./model/wiki.model")

    # print(sys.stdout.buffer.write((args.positive).encode('utf-8')))
    results = model.wv.most_similar(positive=["好き", "嫌い", "花"])
    print("入力：好き，嫌い，花")
    print("候補：")
    for result in results:
        print(result)

    print("\nランダムに選択された3単語")
    print(random.sample(results, 3))
