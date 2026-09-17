"""Conservative document admission for prose boundary training.

This selects intact documents; it neither edits text nor labels boundaries.
Structured material is quarantined for later extraction, not judged incorrect.
"""
import re
import unicodedata
from collections import Counter

VERSION = "continuous-prose-v2"
POLICY = {
    "version": VERSION,
    "minimum_han_characters": 80,
    "minimum_han_share_of_han_and_latin": 0.70,
    "minimum_sentence_units": 2,
    "minimum_han_per_sentence_unit": 8,
    "minimum_terminated_line_sentence_han": 12,
    "minimum_prose_han_share": 0.40,
    "regular_short_clause_minimum": 12,
    "regular_short_clause_han_share": 0.65,
    "minimum_score": 0.5,
    "source_allowlist": None,
    "scope": "Intact continuous-prose candidates, not certified correct training labels",
}
HAN_PATTERN = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]"
HAN = re.compile(HAN_PATTERN)
NON_HAN = re.compile("[^" + HAN_PATTERN[1:-1] + "]+")
LATIN = re.compile(r"[A-Za-z]")
NUMBER = r"(?:\d{1,4}(?:\.\d{1,3}){0,4}|[一二三四五六七八九十百]+)"
# Preserve source newlines for structure detection. Dates, decimal numbers,
# versions, percentages and ordinary inline quantities are not list headings.
LINE_NUMBER = re.compile(r"(?:^|\n)[ \t]*(?:" + NUMBER + r"[.、,):\-]|[（(]" + NUMBER + r"[)）])(?=\s|" + HAN_PATTERN + r"|[（(\"“])")
INLINE_NUMBER = re.compile(r"(?<![\w.])" + NUMBER + r"[.、,)](?=\s*" + HAN_PATTERN + r")")
DENSE_ARABIC_NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d{1,3}\.(?=\s*" + HAN_PATTERN + r")")
CHAPTER = re.compile(r"(?:^|\s)第[一二三四五六七八九十百\d]+[章节篇部]|§\s*\d+")
OPTIONS = re.compile(r"(?<![A-Za-z])[A-D][.、:)][ \t]*(?=" + HAN_PATTERN + r"|\d)")
BLANK = re.compile(r"\([ \t]*\)|_{3,}")
EXAM = re.compile(r"单项选择题|多项选择题|填空题|判断题|每小题.{0,8}分|正确答案|参考答案|试题解析")
MARKUP = re.compile(r"\$P\$|\[PAR\]|\[SEP\]|<\|[^>\n]{1,40}\|>|&(?:nbsp|amp|lt|gt);|</?[A-Za-z][^>\n]{0,200}>")
CODE = re.compile(r"```|#include\s*[<\"]|\b(?:def|function)\s+\w+\s*\(|\b(?:var|const|let)\s+\w+\s*=|\bSELECT\s+.+\bFROM\b",re.I)
MATH = re.compile(r"\\(?:frac|sum|sqrt|begin|end)\b|[A-Za-z\d)）]\s*[=<>±×÷^]\s*[A-Za-z\d(（]|\$[^\n$]{1,150}\$")
TABLE_ROW = re.compile(r"(?:\S[ \t]*\|){3,}|(?:\S\t){3,}")
CATALOG_CUE = re.compile(r"图书目录|章节目录|目录一览|内容目录|参考文献|目\s*录")
BOILERPLATE = re.compile(r"上一篇|下一篇|相关阅读|相关推荐|往期精选|网站地图|友情链接|立即购买|加入购物车|在线客服|咨询热线")
SENTENCE_END = re.compile(r"[。!?;]|(?<=[^\d])\.(?=[^\d]|$)")
GLOSSARY_CODE = re.compile(r"(?:^|\n)[ \t]*[A-Z]{1,3}\.\s*\d{1,4}(?=\s|" + HAN_PATTERN + r")")
URL = re.compile(r"https?://\S+|www\.\S+",re.I)
LINE_SENTENCE = re.compile(r"[^。!?;\n]+(?:[。!?;]|(?<=[^\d])\.(?=[^\d]|$))")
REASON_BITS = {name:1 << i for i,name in enumerate([
    "insufficient_chinese", "mixed_language", "insufficient_prose",
    "numbered_structure", "catalog", "exercise", "extraction_noise",
    "code_or_formula", "table", "boilerplate_dominated", "invalid_score", "below_score_floor",
    "fragmented_or_verse", "dense_enumeration",
])}


def inspect_document(raw, score=None):
    """Return admission reasons and inspectable counts; leave the input intact."""
    text = unicodedata.normalize("NFKC", str(raw or ""))
    han = len(HAN.findall(text)); latin = len(LATIN.findall(text))
    line_numbers = len(LINE_NUMBER.findall(text))
    inline_numbers = len(INLINE_NUMBER.findall(text))
    dense_numbers = len(DENSE_ARABIC_NUMBER.findall(text))
    chapters = len(CHAPTER.findall(text))
    options = len(OPTIONS.findall(text)); blanks = len(BLANK.findall(text))
    exams = len(EXAM.findall(text)); markup = len(MARKUP.findall(text))
    math = len(MATH.findall(text)); code = len(CODE.findall(text))
    tables = sum(bool(TABLE_ROW.search(line)) for line in text.splitlines())
    boilerplate = len(BOILERPLATE.findall(text))
    prose_text=URL.sub('',text)
    prose = sum(len(HAN.findall(part)) >= POLICY['minimum_han_per_sentence_unit']
                for part in SENTENCE_END.split(prose_text))
    prose_han=sum(count for match in LINE_SENTENCE.finditer(prose_text)
                  if (count:=len(HAN.findall(match.group()))) >= POLICY['minimum_terminated_line_sentence_han'])
    glossary_codes=len(GLOSSARY_CODE.findall(text))
    short_units=[len(HAN.findall(part)) for part in re.split(r'[,。!?;\n]',prose_text)]
    regular_short=[count for count in short_units if count in (5,7)]
    reasons=[]
    if han < POLICY['minimum_han_characters']: reasons.append('insufficient_chinese')
    if han / max(1,han+latin) < POLICY['minimum_han_share_of_han_and_latin']: reasons.append('mixed_language')
    if prose < POLICY['minimum_sentence_units']: reasons.append('insufficient_prose')
    if line_numbers or inline_numbers >= 2 or dense_numbers >= 4 or glossary_codes>=2: reasons.append('numbered_structure')
    if chapters >= 3 or (CATALOG_CUE.search(text) and line_numbers+inline_numbers+dense_numbers+chapters >= 3): reasons.append('catalog')
    if options >= 3 or (blanks >= 2 and (exams or line_numbers+dense_numbers >= 2)) or exams >= 3: reasons.append('exercise')
    if markup >= 2 or text.count('\ufffd') >= 2: reasons.append('extraction_noise')
    if code or math >= 3: reasons.append('code_or_formula')
    if tables >= 3: reasons.append('table')
    if boilerplate >= 5 and han/max(1,boilerplate) < 100: reasons.append('boilerplate_dominated')
    if (prose_han/max(1,han)<POLICY['minimum_prose_han_share']
            or (len(regular_short)>=POLICY['regular_short_clause_minimum']
                and sum(regular_short)/max(1,han)>=POLICY['regular_short_clause_han_share'])):
        reasons.append('fragmented_or_verse')
    if text.count('、')>=15 and text.count('、')/max(1,han)>=.045:reasons.append('dense_enumeration')
    try:
        numeric_score=float(score)
        if not 0 <= numeric_score <= 1: raise ValueError('score range')
    except (ValueError,TypeError):
        numeric_score=None; reasons.append('invalid_score')
    if numeric_score is not None and numeric_score < POLICY['minimum_score']: reasons.append('below_score_floor')
    return {'accepted':not reasons,'reasons':reasons,'reason_mask':sum(REASON_BITS[r] for r in reasons),
            'han':han,'latin':latin,'sentence_units':prose,'line_numbers':line_numbers,
            'inline_numbers':inline_numbers,'dense_numbers':dense_numbers,'chapters':chapters,
            'options':options,'blanks':blanks,'markup':markup,'math':math,'code':code,
            'table_rows':tables,'score':numeric_score,'prose_han':prose_han,'glossary_codes':glossary_codes}


SPAN_POLICY = {
    'version':'continuous-prose-spans-v1',
    'minimum_han_characters':60,
    'minimum_long_clauses':2,
    'minimum_han_per_long_clause':8,
    'minimum_long_clause_han_share':0.70,
    'joining':'never join across physical source lines or rejected spans',
    'text_mutation':'none; indices are Unicode code-point offsets in the raw document',
}
CALL_TO_ACTION = re.compile(
    r'点击(?:下方|上方|这里|阅读原文)|长按.{0,12}(?:识别|关注)|扫描.{0,30}二维码|'
    r'(?:扫码|扫描).{0,12}(?:领取|关注|咨询)|(?:关注|回复).{0,15}(?:公众号|领取)|'
    r'如有侵权|版权所有|视频加载中|往期(?:推荐|精选)|转载(?:请|需)|'
    r'联系电话|咨询热线|联系方式|微信号[:：]|[“"]写留言[”"]')


def select_prose_spans(raw,score):
    """Select source lines that contain sustained prose; never concatenate them."""
    selected=[];rejected=Counter()
    for match in re.finditer(r'[^\r\n]+',raw):
        value=match.group();stripped=value.strip()
        if not stripped:continue
        start=match.start()+len(value)-len(value.lstrip());end=match.end()-(len(value)-len(value.rstrip()))
        normalized=unicodedata.normalize('NFKC',stripped)
        normalized_for_counts=URL.sub('',normalized)
        assessment=inspect_document(stripped,score)
        reasons=[r for r in assessment['reasons'] if r not in ['insufficient_chinese','insufficient_prose','fragmented_or_verse']]
        han=assessment['han']
        if han<SPAN_POLICY['minimum_han_characters']:reasons.append('short_line')
        clauses=[len(HAN.findall(part)) for part in re.split(r'[,。!?;]|(?<!\d)\.(?!\d)',normalized_for_counts)]
        long=[count for count in clauses if count>=SPAN_POLICY['minimum_han_per_long_clause']]
        if len(long)<SPAN_POLICY['minimum_long_clauses'] or sum(long)/max(1,han)<SPAN_POLICY['minimum_long_clause_han_share']:
            reasons.append('not_sustained_prose')
        if CALL_TO_ACTION.search(normalized):reasons.append('call_to_action_or_boilerplate')
        letters=''.join(HAN.findall(normalized))
        if len(letters)>=60:
            repeated=Counter(letters[i:i+4] for i in range(len(letters)-3))
            maximum=max(repeated.values(),default=0)
            if maximum>=8 and maximum*4/len(letters)>=.20:reasons.append('repetitive_text')
        if reasons:rejected.update(set(reasons))
        else:
            assert raw[start:end]==stripped
            selected.append({'start':start,'end':end,'han':han,'long_clauses':len(long)})
    return selected,dict(rejected)
