"""Task-level checks for conservative prose admission, independent of proxies."""
import unittest
from web_document_quality import inspect_document, select_prose_spans

PROSE = ('昨天傍晚我们沿着河边散步，看到附近的居民正在整理花园。'
         '这座城市每年都会安排工作人员检查步道，希望为大家提供舒适的环境。'
         '研究小组随后记录了植物的生长情况，并向社区介绍观察到的变化。'
         '大家可以按照自己的时间参加活动，也可以在家中阅读相关的说明。')


class ProseAdmissionTests(unittest.TestCase):
    def test_keeps_narrative_and_mixed_context_without_changing_it(self):
        for text in [PROSE, PROSE.replace('，', ',').replace('。','.'),
                     PROSE+'会议定于2026年9月16日8:30开始，设备版本是2.3.1，结果为98.7%。',
                     '新闻报道讨论了今年考试的变化。'+PROSE,
                     PROSE+'老师说：“欢迎大家明天再来。”',
                     PROSE+'“清单”也是这次讨论的话题，我们没有因此改变活动安排。']:
            self.assertTrue(inspect_document(text,.7)['accepted'],text)

    def test_quarantines_structured_documents_even_at_high_scores(self):
        fixtures={
            'numbered_structure': PROSE+'\n1.第一部分\n2.第二部分',
            'catalog': PROSE+' 图书目录 一,理论 §1基本原理 §2常见现象 §3研究方法',
            'exercise': PROSE+'\n下列说法正确的是()。A.提高 B.降低 C.不变 D.不能确定',
            'extraction_noise': PROSE+'$P$正文$P$正文',
            'code_or_formula': PROSE+'\n```python\ndef calculate(x): return x\n```',
        }
        for reason,text in fixtures.items():
            result=inspect_document(text,.99)
            self.assertFalse(result['accepted']);self.assertIn(reason,result['reasons'])

    def test_not_a_quality_score_only_filter(self):
        self.assertTrue(inspect_document(PROSE,.51)['accepted'])
        self.assertFalse(inspect_document('非常短的标题',.99)['accepted'])
        self.assertIn('invalid_score',inspect_document(PROSE,'nan')['reasons'])

    def test_keeps_dates_but_quarantines_non_prose_structures(self):
        self.assertTrue(inspect_document('2026-09-16\n'+PROSE,.7)['accepted'])
        self.assertIn('numbered_structure',inspect_document(PROSE+'\n1-学校东门\n2-图书馆北门',.9)['reasons'])
        self.assertIn('numbered_structure',inspect_document('P.11 名词解释\n'+PROSE+'\nP.12 另一词条\n'+PROSE,.9)['reasons'])
        verse='春风又绿江南岸，明月何时照我还。\n'*10
        self.assertIn('fragmented_or_verse',inspect_document(verse,.99)['reasons'])

    def test_extracts_intact_spans_without_joining_across_noise(self):
        footer='点击下方公众号，回复关键词即可领取。'+PROSE
        raw='标题\n  '+PROSE+'  \n'+footer+'\n\n'+PROSE
        spans,reasons=select_prose_spans(raw,.8)
        self.assertEqual([raw[s['start']:s['end']] for s in spans],[PROSE,PROSE])
        self.assertGreater(spans[1]['start'],spans[0]['end'])
        self.assertEqual(reasons['call_to_action_or_boilerplate'],1)


if __name__=='__main__': unittest.main()
