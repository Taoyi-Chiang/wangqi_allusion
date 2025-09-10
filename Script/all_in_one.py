import csv
import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# 1. 腳本配置區 - 所有參數都在這裡設定
# ============================================================================
class Config:
    # 若只想處理特定「賦家」，請填寫名字；若想全部不過濾，可設為空字串 ""
    AUTHOR_NAME = "王起"

    # 檔案路徑設定
    SENTENCE_MATCH_JSON_PATH = Path(r"D:/wangqi_allusion/output/origin_text.json")
    MAIN_JSON_PATH           = Path(r"D:/wangqi_allusion/output/manual_origin_text_ckip.json")
    COMPARED_TEXT_PATH       = Path(r"D:/lufu_allusion/data/raw/compared_text/十三經")
    OUTPUT_CSV_PATH          = Path(r"D:/wangqi_allusion/output/direct_allusion.csv")

    # 比對參數
    CHARS_TO_REMOVE = "﹔。，、：；！？（）〔〕「」[]『』《》〈〉\\#\\-\\－\\(\\)\\[\\]\\]\\\\/ ,.:;!?~1234567890¶"
    NGRAM_RANGE     = [2, 3, 4]

    # 排除字串清單 (詞級比對用)
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
    EXCLUDE_TOKENS = set(PREFIX_EXCLUDE + SUFFIX_EXCLUDE)

# ============================================================================
# 2. 核心功能函數
# ============================================================================
def normalize(text: str) -> str:
    """統一化文字格式。"""
    return unicodedata.normalize("NFKC", text)

def extract_char_ngrams(chars: list, n: int) -> list:
    """從字元列表中提取 N-gram。"""
    return ["".join(chars[i:i+n]) for i in range(len(chars) - n + 1)]

def load_and_process_data():
    """載入主資料與句級比對結果，並預先處理。"""
    print("🚀 正在載入與處理資料...")

    # 載入句級比對結果，建立已匹配句子的集合
    try:
        sentence_match_list = json.load(open(Config.SENTENCE_MATCH_JSON_PATH, "r", encoding="utf-8"))
    except FileNotFoundError:
        print(f"⚠️ 找不到檔案: {Config.SENTENCE_MATCH_JSON_PATH}，將跳過句級比對。")
        sentence_match_list = []
    
    sentence_matched_keys = set()
    for entry in sentence_match_list:
        entry_author = entry.get("author", "").strip()
        if Config.AUTHOR_NAME and entry_author != Config.AUTHOR_NAME:
            continue
        key = (entry_author, entry.get("article_title", "").strip(), 
               str(entry.get("paragraph_num", "")).strip(), str(entry.get("group_num", "")).strip(), 
               str(entry.get("sentence_num", "")).strip())
        sentence_matched_keys.add(key)

    # 載入主 JSON 檔案，建立查找字典
    main_data = json.load(open(Config.MAIN_JSON_PATH, "r", encoding="utf-8"))
    main_lookup = {}
    sentence_order_map = {}
    hash_to_terms = {}
    
    order_counter = 0
    term_order = 0

    for essay in main_data:
        author = str(essay.get("賦家", "")).strip()
        if Config.AUTHOR_NAME and author != Config.AUTHOR_NAME:
            continue
        article_title = str(essay.get("賦篇", "")).strip()
        art_no = str(essay.get("篇號", "")).strip()

        for para in essay.get("段落", []):
            para_no = str(para.get("段落編號", "")).strip()
            for grp in para.get("句組", []):
                grp_no = str(grp.get("句組編號", "")).strip()
                for sentence in grp.get("句子", []):
                    sent_no = str(sentence.get("句編號", "")).strip()
                    # *** 重要修正: 從 "內容" 欄位讀取原始句 ***
                    orig_sent = sentence.get("內容", "").strip() 

                    key_sentence = (author, article_title, para_no, grp_no, sent_no)
                    main_lookup[key_sentence] = (art_no, orig_sent)
                    sentence_order_map[key_sentence] = order_counter
                    order_counter += 1

                    if key_sentence in sentence_matched_keys:
                        continue

                    token_list = sentence.get("tokens", [])
                    if not isinstance(token_list, list):
                        continue

                    for token in token_list:
                        tok = token.strip()
                        if not tok or tok in Config.EXCLUDE_TOKENS:
                            continue
                        
                        token_hash = hash(tok) & 0x7FFFFFFFFFFFFFFF
                        term_info = {
                            "order": term_order, "篇號": art_no, "賦家": author, "賦篇": article_title,
                            "段落編號": para_no, "句組編號": grp_no, "句編號": sent_no,
                            "原始句": orig_sent, "tokens": tok
                        }
                        hash_to_terms.setdefault(token_hash, []).append(term_info)
                        term_order += 1

    all_term_hashes = set(hash_to_terms.keys())
    return sentence_match_list, sentence_matched_keys, main_lookup, sentence_order_map, hash_to_terms, all_term_hashes

def perform_token_matching(hash_to_terms, all_term_hashes, sentence_matched_keys, main_lookup, sentence_order_map):
    """執行詞級比對，產生結果。"""
    print("🔍 正在進行詞級比對...")
    token_level_matches = {}
    all_txt_files = list(Config.COMPARED_TEXT_PATH.rglob("*.txt"))

    for fp in tqdm(all_txt_files, desc="掃描比對檔案", unit="file"):
        raw = normalize(fp.read_text(encoding='utf-8')).replace("\n", "")
        raw = re.sub(r"<[^>]*>", "", raw)
        segments = re.split(f"[{re.escape(Config.CHARS_TO_REMOVE)}]", raw)

        for idx, seg in enumerate(segments):
            seg = normalize(seg).strip()
            if not seg or len(seg) < min(Config.NGRAM_RANGE):
                continue
            
            seg_hashes = set()
            chars = list(seg)
            for n in Config.NGRAM_RANGE:
                for gram in extract_char_ngrams(chars, n):
                    g_hash = hash(gram) & 0x7FFFFFFFFFFFFFFF
                    if g_hash in all_term_hashes:
                        seg_hashes.add(g_hash)

            for term_hash in seg_hashes:
                for term_info in hash_to_terms[term_hash]:
                    key_sentence = (term_info["賦家"], term_info["賦篇"], term_info["段落編號"],
                                    term_info["句組編號"], term_info["句編號"])

                    if key_sentence in sentence_matched_keys:
                        continue
                    
                    art_no, orig_sentence = main_lookup.get(key_sentence, ("", ""))
                    
                    match_dict = {
                        "sentence_order": sentence_order_map.get(key_sentence, float('inf')),
                        "order": term_info["order"], "篇號": art_no, "賦家": term_info["賦家"],
                        "賦篇": term_info["賦篇"], "段落編號": term_info["段落編號"],
                        "句組編號": term_info["句組編號"], "句編號": term_info["句編號"],
                        "原始句": orig_sentence, "tokens": term_info["tokens"],
                        "matched_file": f"{fp.parent.name}\\{fp.stem}",
                        "matched_index": idx, "matched": seg, "similarity": "NA"
                    }
                    token_level_matches.setdefault(key_sentence, []).append(match_dict)
    
    return token_level_matches

def generate_and_save_csv(matches):
    """將所有比對結果寫入 CSV 檔案。"""
    csv_headers = [
        "篇號", "賦家", "賦篇", "段落編號", "句組編號", "句編號", "原始句", "tokens",
        "matched_file", "matched_index", "matched", "similarity"
    ]

    Config.OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        for m in matches:
            writer.writerow({key: m.get(key, "") for key in csv_headers})

    print(f"✅ 完成！共 {len(matches)} 筆結果，已存到 {Config.OUTPUT_CSV_PATH}")

def main():
    """主執行函數。"""
    sentence_match_list, sentence_matched_keys, main_lookup, sentence_order_map, hash_to_terms, all_term_hashes = load_and_process_data()
    token_level_matches = perform_token_matching(hash_to_terms, all_term_hashes, sentence_matched_keys, main_lookup, sentence_order_map)

    # 合併所有比對結果
    all_matches = []
    sorted_keys = sorted(sentence_order_map.keys(), key=lambda k: sentence_order_map[k])

    for key_sentence in sorted_keys:
        # 將解包的程式碼移到這裡，在任何 'continue' 語句之前
        author, article_title, para_no, grp_no, sent_no = key_sentence

        # 現在篩選條件可以放在這裡，因為變數已經被定義了
        if Config.AUTHOR_NAME and author != Config.AUTHOR_NAME:
            continue
        
        art_no, orig_sent = main_lookup.get(key_sentence, ("", ""))

        if key_sentence in sentence_matched_keys:
            for entry in sentence_match_list:
                k = (entry.get("author", "").strip(), entry.get("article_title", "").strip(), 
                     str(entry.get("paragraph_num", "")).strip(), str(entry.get("group_num", "")).strip(), 
                     str(entry.get("sentence_num", "")).strip())
                if k == key_sentence:
                    all_matches.append({
                        "篇號": str(entry.get("article_num", "")).strip(), "賦家": entry.get("author", ""),
                        "賦篇": entry.get("article_title", ""), "段落編號": str(entry.get("paragraph_num", "")).strip(),
                        "句組編號": str(entry.get("group_num", "")).strip(), "句編號": str(entry.get("sentence_num", "")).strip(),
                        "原始句": entry.get("original", ""), "tokens": entry.get("original", ""),
                        "matched_file": entry.get("matched_file", ""), "matched_index": entry.get("matched_index", ""),
                        "matched": entry.get("matched", ""), "similarity": str(entry.get("similarity", ""))
                    })
            continue

        if key_sentence in token_level_matches:
            # 在這裡，我們從 main_lookup 重新獲取完整的原始句，而不是依賴 token_level_matches 裡的舊值
            art_no, orig_sent_from_main = main_lookup.get(key_sentence, ("", ""))
            
            sorted_token_matches = sorted(token_level_matches[key_sentence], key=lambda x: x["order"])
            
            for ref in sorted_token_matches:
                all_matches.append({
                    "篇號": art_no, 
                    "賦家": author, 
                    "賦篇": article_title,
                    "段落編號": para_no, 
                    "句組編號": grp_no, 
                    "句編號": sent_no,
                    "原始句": orig_sent_from_main,  # <-- 使用從 main_lookup 讀取到的完整原始句
                    "tokens": ref["tokens"],
                    "matched_file": ref["matched_file"], 
                    "matched_index": ref["matched_index"],
                    "matched": ref["matched"], 
                    "similarity": ref["similarity"]
                })
            continue

        all_matches.append({
            "篇號": art_no, "賦家": author, "賦篇": article_title, "段落編號": para_no,
            "句組編號": grp_no, "句編號": sent_no, "原始句": orig_sent,
            "tokens": "", "matched_file": "", "matched_index": "", "matched": "", "similarity": ""
        })

    generate_and_save_csv(all_matches)

if __name__ == "__main__":
    main()