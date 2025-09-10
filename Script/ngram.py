import csv
import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm

# === 可自訂參數 ===
# 若只想處理特定「賦家」，請填寫名字；若想全部不過濾，可設為空字串 ""
author_name = "王起"

# === 路徑設定（範例）===
SENTENCE_MATCH_JSON_PATH = Path(r"D:/wangqi_allusion/output/sentence_allusion.json")
MAIN_JSON_PATH           = Path(r"D:/wangqi_allusion/output/manual_origin_text_ckip.json")
COMPARED_TEXT_PATH       = Path(r"D:/lufu_allusion/data/raw/compared_text/十三經")
OUTPUT_CSV_PATH          = Path(r"D:/wangqi_allusion/output/direct_allusion.csv")

# 用於切句的字元
CHARS_TO_REMOVE = "﹔。，、：；！？（）〔〕「」[]『』《》〈〉\\#\\-\\－\\(\\)\\[\\]\\]\\\\/ ,.:;!?~1234567890¶"
NGRAM_RANGE     = [2, 3, 4]

# === 新增：排除字串清單 ===
PREFIX_EXCLUDE = [
    "徒觀其", "矞夫", "矞乃", "至夫", "懿夫", "蓋由我君", "重曰", "是知", "嗟夫", "夫其", "懿其", "所以",
    "想夫", "其始也", "當其", "況復", "時則", "至若", "豈獨", "若乃", "今則", "乃知", "既而", "嗟乎",
    "故我后", "觀夫", "然而", "爾乃", "是以", "原夫", "曷若", "斯則", "於時", "方今", "亦何必", "若然",
    "客有", "至於", "則知", "且夫", "斯乃", "況", "於是", "覩夫", "且彼", "豈若", "已而", "始也", "故",
    "然則", "豈如我", "豈不以", "我國家", "其工者", "所謂", "今吾君", "及夫", "爾其", "將以", "可以", "今",
    "國家", "然後", "向非我后", "則有", "彼", "惜乎", "由是", "乃言曰", "若夫", "亦何用", "不然",
    "嘉其", "今則", "徒美夫", "故能", "有探者曰", "惜如", "而況", "逮夫", "誠夫", "於戲", "洎乎", "伊昔",
    "則將", "今則", "況今", "士有", "暨乎", "亦何辨夫", "俾夫", "亦猶", "瞻夫", "時也", "固知", "足以",
    "矞國家", "比乎", "亦由", "觀其", "將俾乎", "聖人", "君子", "於以", "乃", "斯蓋", "噫", "夫惟",
    "高皇帝", "帝既", "嘉其", "始則", "又安得", "其", "儒有", "當是時也", "夫然", "宜乎", "故其", "國家",
    "爾其始也", "今我國家", "是時", "有司", "向若", "我皇", "故王者", "則", "鄒子", "孰", "暨夫", "用能",
    "故將", "況其", "故宜", "王者", "聖上", "先王", "乃有", "況乃", "別有", "今者", "固宜", "皇上", "且其",
    "徒觀夫", "帝堯以", "始其", "倏而", "乃曰", "向使", "漢武帝", "先是", "他日", "乃命", "觀乎", "國家以",
    "墨子", "借如", "足以", "上乃", "嗚呼", "昔伊", "先賢", "遂使", "豈比夫", "固其", "況有", "魯恭王", "皇家",
    "吾君是時", "知", "周穆王", "則有", "是用", "乃言曰", "及", "故夫", "矞乎", "夫以", "寧令", "如", "然則",
    "滅明乃", "遂", "悲夫", "安得", "故得", "且見其", "是何", "莫不", "士有", "知其", "未若", "蓋以", "固可以",
    "豈徒", "豈比夫", "是故"
]
SUFFIX_EXCLUDE = [
    "曰", "哉", "矣", "也", "矣哉", "乎", "焉", "者也", "也矣哉"
]

# 合併所有需要排除的詞彙，建立一個集合以提高查詢效率
EXCLUDE_TOKENS = set(PREFIX_EXCLUDE + SUFFIX_EXCLUDE)

def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def extract_char_ngrams(chars: list, n: int) -> list:
    return ["".join(chars[i:i+n]) for i in range(len(chars)-n+1)]

# ----------------------------------------------------------------------------
# 0. 讀取「句級比對結果」JSON，建立一個集合：key = (author, article_title, paragraph_num, group_num, sentence_num)
#    只蒐集符合 author_name 的，如果 author_name 為空則不過濾任何作者
# ----------------------------------------------------------------------------
sentence_match_list = json.load(open(SENTENCE_MATCH_JSON_PATH, "r", encoding="utf-8"))
sentence_matched_keys = set()

for entry in sentence_match_list:
    entry_author = entry.get("author", "").strip()
    if author_name and entry_author != author_name:
        continue

    key = (
        entry_author,
        entry.get("article_title", "").strip(),
        str(entry.get("paragraph_num", "")).strip(),
        str(entry.get("group_num", "")).strip(),
        str(entry.get("sentence_num", "")).strip()
    )
    sentence_matched_keys.add(key)

# ----------------------------------------------------------------------------
# 1. 讀主 JSON，建立 main_lookup 及 sentence_order_map，並且同時把每句的 tokens 轉成 hash_to_terms
#    若該句已在句級比對中命中，則跳過不加入 hash_to_terms
# ----------------------------------------------------------------------------
main_data = json.load(open(MAIN_JSON_PATH, "r", encoding="utf-8"))

main_lookup = {}        # key_sentence → (篇號, 原始句)
sentence_order_map = {} # key_sentence → 全局排序 index
hash_to_terms = {}      # token_hash → list of term_info
all_term_hashes = set()

order_counter = 0
term_order = 0

for essay in main_data:
    author          = str(essay.get("賦家", "")).strip()
    if author_name and author != author_name:
        continue

    article_title = str(essay.get("賦篇", "")).strip()
    art_no        = str(essay.get("篇號", "")).strip()

    for para in essay.get("段落", []):
        para_no = str(para.get("段落編號", "")).strip()
        for grp in para.get("句組", []):
            grp_no = str(grp.get("句組編號", "")).strip()
            for sentence in grp.get("句子", []):
                sent_no    = str(sentence.get("句編號", "")).strip()
                orig_sent  = sentence.get("原始句", "").strip() # <--- 修正: 改為讀取 "原始句"

                key_sentence = (author, article_title, para_no, grp_no, sent_no)
                main_lookup[key_sentence] = (art_no, orig_sent)
                sentence_order_map[key_sentence] = order_counter
                order_counter += 1

                # 若這句已在句級比對結果命中，跳過不把它的 tokens 納入 hash_to_terms
                if key_sentence in sentence_matched_keys:
                    continue

                # 從 JSON 拿出 tokens
                token_list = sentence.get("tokens", [])
                if not isinstance(token_list, list):
                    continue

                for token in token_list:
                    tok = token.strip()
                    if not tok or tok in EXCLUDE_TOKENS:
                        continue

                    token_hash = hash(tok) & 0x7FFFFFFFFFFFFFFF
                    term_info = {
                        "order":         term_order,
                        "篇號":          art_no,
                        "賦家":          author,
                        "賦篇":          article_title,
                        "段落編號":      para_no,
                        "句組編號":      grp_no,
                        "句編號":        sent_no,
                        "原始句":        orig_sent,
                        "tokens":        tok
                    }
                    hash_to_terms.setdefault(token_hash, []).append(term_info)
                    term_order += 1

all_term_hashes = set(hash_to_terms.keys())

# ----------------------------------------------------------------------------
# 2. 掃描比較文本，對尚未句級 match 的句子做 n-gram 比對
#    若某句在 sentence_matched_keys 中，就跳過不做詞級比對
# ----------------------------------------------------------------------------
token_level_matches = {}

all_txt_files = list(COMPARED_TEXT_PATH.rglob("*.txt"))
for fp in tqdm(all_txt_files, desc="Scanning files", unit="file"):
    raw = normalize(fp.read_text(encoding='utf-8')).replace("\n", "")
    raw = re.sub(r"<[^>]*>", "", raw)
    segments = re.split(f"[{re.escape(CHARS_TO_REMOVE)}]", raw)

    for idx, seg in enumerate(segments):
        seg = normalize(seg).strip()
        if not seg or len(seg) < min(NGRAM_RANGE):
            continue

        # 計算該段所有 n-gram 的 hash
        seg_hashes = set()
        chars = list(seg)
        for n in NGRAM_RANGE:
            for gram in extract_char_ngrams(chars, n):
                g_hash = hash(gram) & 0x7FFFFFFFFFFFFFFF
                if g_hash in all_term_hashes:
                    seg_hashes.add(g_hash)

        # 每個符合的 token_hash，都可能對應到多筆 term_info
        for term_hash in seg_hashes:
            for term_info in hash_to_terms[term_hash]:
                key_sentence = (
                    term_info["賦家"],
                    term_info["賦篇"],
                    term_info["段落編號"],
                    term_info["句組編號"],
                    term_info["句編號"]
                )

                # 若已在句級 match，就跳過詞級 match
                if key_sentence in sentence_matched_keys:
                    continue

                # 提取完整句子資訊
                art_no, orig_sentence = main_lookup.get(key_sentence, ("", ""))

                match_dict = {
                    "sentence_order":  sentence_order_map.get(key_sentence, 10**9),
                    "order":           term_info["order"],
                    "篇號":            art_no,
                    "賦家":            term_info["賦家"],
                    "賦篇":            term_info["賦篇"],
                    "段落編號":        term_info["段落編號"],
                    "句組編號":        term_info["句組編號"],
                    "句編號":          term_info["句編號"],
                    "原始句":          orig_sentence,
                    "tokens":          term_info["tokens"],
                    "matched_file":    f"{fp.parent.name}\\{fp.stem}",
                    "matched_index":   idx,
                    "matched":         seg,
                    "similarity":      "NA"
                }

                token_level_matches.setdefault(key_sentence, []).append(match_dict)


# ----------------------------------------------------------------------------
# 3. 合併所有句：若有句級 match → 先列句級 match；否則若有 token-level match → 按 order 排序列出；否則空白列
# ----------------------------------------------------------------------------
matches = []

sorted_keys = sorted(
    sentence_order_map.keys(),
    key=lambda k: sentence_order_map[k]
)

for key_sentence in sorted_keys:
    author, article_title, para_no, grp_no, sent_no = key_sentence
    # 若指定 author_name 而這句不是同一作者，就跳過
    if author_name and author != author_name:
        continue

    art_no, orig_sent = main_lookup.get(key_sentence, ("", ""))

    # (1) 句級 match
    if key_sentence in sentence_matched_keys:
        # 把所有對應的句級比對結果列出來
        for entry in sentence_match_list:
            k = (
                entry.get("author", "").strip(),
                entry.get("article_title", "").strip(),
                str(entry.get("paragraph_num", "")).strip(),
                str(entry.get("group_num", "")).strip(),
                str(entry.get("sentence_num", "")).strip()
            )
            if k == key_sentence:
                matches.append({
                    "篇號":          entry.get("article_num", ""),
                    "賦家":          entry.get("author", ""),
                    "賦篇":          entry.get("article_title", ""),
                    "段落編號":      str(entry.get("paragraph_num", "")).strip(),
                    "句組編號":      str(entry.get("group_num", "")).strip(),
                    "句編號":        str(entry.get("sentence_num", "")).strip(),
                    "原始句":        entry.get("original", ""),
                    "tokens":        entry.get("original", ""),
                    "matched_file":    entry.get("matched_file", ""),
                    "matched_index":   entry.get("matched_index", ""),
                    "matched":         entry.get("matched", ""),
                    "similarity":      str(entry.get("similarity", ""))
                })
        continue

    # (2) token-level match（按 order 排序）
    if key_sentence in token_level_matches:
        sorted_token_matches = sorted(
            token_level_matches[key_sentence],
            key=lambda x: x["order"]
        )
        for ref in sorted_token_matches:
            matches.append({
                "篇號":          ref["篇號"],
                "賦家":          ref["賦家"],
                "賦篇":          ref["賦篇"],
                "段落編號":      ref["段落編號"],
                "句組編號":      ref["句組編號"],
                "句編號":        ref["句編號"],
                "原始句":        ref["原始句"],
                "tokens":        ref["tokens"],
                "matched_file":    ref["matched_file"],
                "matched_index":   ref["matched_index"],
                "matched":         ref["matched"],
                "similarity":      ref["similarity"]
            })
        continue

    # (3) 無任何 match → 空白列
    matches.append({
        "篇號":          art_no,
        "賦家":          author,
        "賦篇":          article_title,
        "段落編號":      para_no,
        "句組編號":      grp_no,
        "句編號":        sent_no,
        "原始句":        orig_sent,
        "tokens":        "",
        "matched_file":    "",
        "matched_index":   "",
        "matched":         "",
        "similarity":      ""
    })

# ----------------------------------------------------------------------------
# 4. 輸出 CSV，僅包含中文欄位
# ----------------------------------------------------------------------------
csv_headers = [
    "篇號",
    "賦家",
    "賦篇",
    "段落編號",
    "句組編號",
    "句編號",
    "原始句",
    "tokens",
    "matched_file",
    "matched_index",
    "matched",
    "similarity"
]

OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()
    for m in matches:
        writer.writerow({key: m.get(key, "") for key in csv_headers})

print(f"✅ Completed: 共 {len(matches)} 筆結果，已存到 {OUTPUT_CSV_PATH}")
