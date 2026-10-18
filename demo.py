"""文物检索 demo：关键词检索（BM25）+ 向量检索（Embedding）+ 混合检索（RRF）+ 概念距离。

用法:
    uv run python3 demo.py "酒器"
    uv run python3 demo.py --concept 青铜器 瓷器 玉器 书法 酒器 礼器
"""
import argparse
import json
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).parent / "data" / "artifacts.jsonl"
USER_DICT = Path(__file__).parent / "data" / "user_dict.txt"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 中文小模型，CPU 可跑；效果更好可换 BAAI/bge-m3
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："  # bge-zh 系列推荐的查询前缀

STOPWORDS = set("的 了 是 和 与 在 于 为 以 及 其 之 一 有 等 由 而 、 ， 。 “ ” （ ） ； ：".split())


def load_docs():
    docs = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    for d in docs:  # 把结构化字段拼进检索文本，这样“青铜”“西汉”等字段也能被搜到
        d["text"] = f"{d['name']}。{d['period']}。{d['material']}。{d['desc']}"
    return docs


def tokenize(text):
    return [w for w in jieba.lcut(text) if w.strip() and w not in STOPWORDS]


class KeywordIndex:
    def __init__(self, docs):
        if USER_DICT.exists():
            jieba.load_userdict(str(USER_DICT))  # 专有名词词典：防止“后母戊鼎”被切碎
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])

    def search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in order if scores[i] > 0]


class VectorIndex:
    def __init__(self, docs, model_name=MODEL_NAME):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        # normalize_embeddings=True 后，点积 == 余弦相似度
        self.emb = self.model.encode([d["text"] for d in docs], normalize_embeddings=True)

    def encode(self, texts, is_query=False):
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True)

    def search(self, query, k=5):
        q = self.encode([query], is_query=True)[0]
        sims = self.emb @ q
        order = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in order]


def rrf(result_lists, k=60, top=5):
    """Reciprocal Rank Fusion：只看排名不看分数，天然解决 BM25 分数和余弦分数量纲不同的问题。"""
    fused, by_id = {}, {}
    for results in result_lists:
        for rank, (doc, _) in enumerate(results, start=1):
            fused[doc["id"]] = fused.get(doc["id"], 0.0) + 1.0 / (k + rank)
            by_id[doc["id"]] = doc
    ranked = sorted(fused.items(), key=lambda x: -x[1])[:top]
    return [(by_id[i], s) for i, s in ranked]


def show(title, results):
    print(f"\n== {title} ==")
    for doc, score in results:
        print(f"  {score:8.4f}  {doc['name']}（{doc['period']}，{doc['material']}）")


def concept_distance(vindex, terms):
    emb = vindex.encode(terms)  # 概念之间对比，不加查询前缀
    sim = emb @ emb.T
    n = max(len(t) for t in terms)
    pad = lambda t: t + "\u3000" * (n - len(t))  # 用全角空格对齐中文
    print("\n== 概念余弦相似度矩阵（越接近 1 越相近；距离 = 1 - 相似度）==")
    print(pad("") + "".join(" " * (8 - 2 * len(t)) + t for t in terms))
    for t, row in zip(terms, sim):
        print(pad(t) + "".join(f"{v:8.3f}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="古代喝酒用的器物")
    ap.add_argument("--concept", nargs="+", help="计算若干概念两两之间的语义距离")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    docs = load_docs()
    vindex = VectorIndex(docs)
    if args.concept:
        concept_distance(vindex, args.concept)
        return

    kindex = KeywordIndex(docs)
    kw = kindex.search(args.query, args.k)
    vec = vindex.search(args.query, args.k)
    print(f"查询：{args.query}    分词：{tokenize(args.query)}")
    show("关键词 BM25", kw)
    show("向量 Embedding（余弦）", vec)
    show("混合 RRF", rrf([kw, vec], top=args.k))


if __name__ == "__main__":
    main()
