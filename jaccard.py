import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
from ckip_transformers.nlp import CkipWordSegmenter
import cupy as cp  # GPU acceleration with CuPy

# ========== 使用者設定 (一處可控) ==========
PARSED_RESULTS_PATH = Path(r"D:/zicai/origin_text.json")
COMPARED_FOLDER_PATH = Path(r"D:/lufu_allusion/data/raw/compared_text/")
OUTPUT_JSON_PATH = Path(r"D:/zicai/sentence_allusion.json")
# 包含半形空格
CHARS_TO_REMOVE = "﹔。，、：；！？（）〔〕「」[]『』《》〈〉\\#\\-\\－\\(\\)\\[\\]\\]\\\\/ ,.:;!?~1234567890¶"
JACCARD_THRESHOLD = 0.7
BATCH_SIZE = 8192  # 可調整批次大小

# ========== 停用詞設定 ==========
PREFIX_EXCLUDE = [ # 原句若以下列詞彙起始則刪除該詞彙
    "故", "觀夫", "夫如是", "匠人", "徒觀其", "厥若", "昔", "則知", "於是", "凡教", "故王者", "偉夫", "自然", 
    "猗嗟", "則知", "其有", "豈比", 
]
SUFFIX_EXCLUDE = [ # 原句若以下列詞彙結尾則刪除該詞彙
  "者哉"
]

# ======== 工具函式 ========
def clean_sentence(text):
    for prefix in PREFIX_EXCLUDE:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    for suffix in SUFFIX_EXCLUDE:
        if text.endswith(suffix):
            return text[:-len(suffix)].strip()
    return text.strip()

def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

# ======== 讀取與清洗原句資料 ========
def load_parsed_results(path):
    print("🔄 載入原句資料...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for art in data:
        for para in art.get("段落", []):
            for grp in para.get("句組", []):
                for sent in grp.get("句子", []):
                    orig = normalize(sent.get("內容", ""))
                    fm = clean_sentence(orig)
                    if fm:
                        records.append({
                            "article_num":   art.get("篇號"),
                            "author":        art.get("賦家"),
                            "article_title": art.get("賦篇"),
                            "paragraph_num": para.get("段落編號"),
                            "group_num":     grp.get("句組編號"),
                            "sentence_num":  sent.get("句編號"),
                            "original":      orig,
                            "for_matching":  fm
                        })
    print(f"✅ 原句資料載入完成，共 {len(records)} 條。")
    return records

# ======== 讀取與清洗待比對句資料 ========
def load_compared_sentences(folder, chars):
    print("🔄 載入待比對句資料...")
    pattern = f"[{re.escape(chars)}]"
    sents = []
    for fp in folder.rglob("*.txt"):
        raw = normalize(fp.read_text(encoding="utf-8").replace("\n", ""))
        raw = re.sub(r"<[^>]*>", "", raw)
        for idx, seg in enumerate(re.split(pattern, raw)):
            seg = normalize(seg)
            fm = clean_sentence(seg)
            if fm:
                sents.append({
                    "matched_file":  fp.parent.name + "/" + fp.stem,
                    "matched_index": idx,
                    "raw":           seg,
                    "for_matching":  fm
                })
    print(f"✅ 待比對資料載入完成，共 {len(sents)} 條。")
    return sents

# ======== 分詞 ========
def segment_in_batches(sentences, segmenter, batch_size=500, text_type=""):
    print(f"🪚 開始分詞：{text_type}（共 {len(sentences)} 條）...")
    tokens = []
    with tqdm(total=len(sentences), desc=f"分詞 ({text_type})", unit="句") as pbar:
        for i in range(0, len(sentences), batch_size):
            batch = [s["for_matching"] for s in sentences[i:i+batch_size]]
            toks = segmenter(batch, show_progress=False)
            tokens.extend(toks)
            pbar.update(len(batch))
    print(f"✅ 分詞 ({text_type}) 完成，共 {len(tokens)} 條 tokens。")
    return tokens

# ======== 詞彙表建立 ========
def build_vocab(tokens_list):
    print("➡️ 建構詞彙表...")
    vocab = sorted({w for toks in tokens_list for w in toks})
    print(f"✅ 詞彙表大小：{len(vocab)} 個詞。")
    return {w: i for i, w in enumerate(vocab)}

# ======== 向量化 ========
def tokens_to_gpu_matrix(tokens_list, word2idx):
    n, d = len(tokens_list), len(word2idx)
    mat = cp.zeros((n, d), dtype=cp.int8)
    for i, toks in enumerate(tokens_list):
        for w in toks:
            idx = word2idx.get(w)
            if idx is not None:
                mat[i, idx] = 1
    return mat

# ======== 批次 Jaccard 計算 ========
def batch_jaccard_gpu(comp_mat, origin_mat):
    comp_f = comp_mat.astype(cp.float16)
    orig_f = origin_mat.astype(cp.float16)
    inter = comp_f.dot(orig_f.T)
    sc = comp_f.sum(axis=1, keepdims=True)
    so = orig_f.sum(axis=1, keepdims=True).T
    jac = inter / (sc + so - inter + 1e-9)
    return jac.max(axis=1), jac.argmax(axis=1)

# ======== 主程式 ==========
def main():
    origin   = load_parsed_results(PARSED_RESULTS_PATH)
    compared = load_compared_sentences(COMPARED_FOLDER_PATH, CHARS_TO_REMOVE)

    ws = CkipWordSegmenter(device=0, model="bert-base")
    origin_tokens   = segment_in_batches(origin,   ws, batch_size=500, text_type="原始文本")
    compared_tokens = segment_in_batches(compared, ws, batch_size=500, text_type="比對文本")

    word2idx   = build_vocab(origin_tokens)
    origin_mat = tokens_to_gpu_matrix(origin_tokens, word2idx)

    total = len(compared)
    print(f"🧪 開始 Jaccard 匹配，共 {total} 筆待比對（批次大小 {BATCH_SIZE}）。")

    matches = []
    # 單一 tqdm 進度條，每處理一句更新一次
    with tqdm(total=total, desc="Jaccard 匹配", unit="句") as pbar:
        for st in range(0, total, BATCH_SIZE):
            ed = min(st + BATCH_SIZE, total)
            comp_batch_mat = tokens_to_gpu_matrix(compared_tokens[st:ed], word2idx)
            scores, idxs   = batch_jaccard_gpu(comp_batch_mat, origin_mat)
            scores, idxs   = cp.asnumpy(scores), cp.asnumpy(idxs)
            for i, (s, oid) in enumerate(zip(scores, idxs)):
                if s >= JACCARD_THRESHOLD:
                    od = origin[oid]
                    cd = compared[st + i]
                    matches.append({
                        **{k: od[k] for k in [
                            "article_num","author","article_title",
                            "paragraph_num","group_num","sentence_num","original"
                        ]},
                        "matched_file":   cd["matched_file"],
                        "matched_index":  cd["matched_index"],
                        "matched":        cd["raw"],
                        "similarity":     float(s)
                    })
                pbar.update(1)

    print(f"✅ Jaccard 匹配完成，共找到 {len(matches)} 筆結果。")
    print("📄 排序並輸出 JSON 檔案...")
    matches = sorted(
        matches,
        key=lambda x: (
            int(x["article_num"]),
            int(x["paragraph_num"]),
            int(x["group_num"]),
            int(x["sentence_num"])
        )
    )
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()