"""Verify provenance, reserved-document exclusion, deduplication and resumption."""
import json
from pathlib import Path
import tempfile
import unittest

from screen_local_web import (ROOT, digest, merge, screen_file, worker_setup)
from prepare_web_data import document_signature, document_split
import pyarrow as pa
import pyarrow.parquet as pq

PROSE=('昨天傍晚我们沿着河边散步，看到附近的居民正在整理花园。'
       '这座城市每年都会安排工作人员检查步道，希望为大家提供舒适的环境。'
       '研究小组随后记录了植物的生长情况，并向社区介绍观察到的变化。'
       '大家可以按照自己的时间参加活动，也可以在家中阅读相关的说明。')


class LocalScreeningTests(unittest.TestCase):
    def test_complete_document_indices_preserve_holdouts_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'out';output.mkdir()
            examples={'train':[],'test':[]}
            i=0
            while len(examples['train'])<2 or not examples['test']:
                raw=chr(0x5000+i)+PROSE;i+=1
                split=document_split(document_signature(raw),20260915)
                if split in examples:examples[split].append(raw)
            good,held=examples['train'][:2];reserved=examples['test'][0]
            bad=good+'\n1.第一部分\n2.第二部分'
            source=root/'source.parquet'
            pq.write_table(pa.table({'content':[good,good,held,reserved,bad],
                                    'source':['fixture']*5,'score':['0.99']*5}),source,row_group_size=2)
            registry=root/'holdouts.json';registry.write_text(json.dumps([document_signature(held)]))
            worker_setup([registry])
            rule_hash=digest(ROOT/'training/web_document_quality.py')
            job=(0,str(source),None,str(output),'all',rule_hash)
            stats=screen_file(job)
            self.assertEqual(stats['documents'],5)
            self.assertEqual(stats['accepted'],4)
            self.assertEqual(stats['candidates'],2)
            self.assertEqual(stats['holdout_accepted'],2)
            self.assertEqual(stats,screen_file(job))
            result=merge(output,[stats],{'fixture':True})
            self.assertEqual(result['unique_train_candidates'],1)
            self.assertEqual(result['duplicate_candidate_documents'],1)
            row=pq.read_table(output/'train-candidates.parquet').to_pylist()[0]
            self.assertEqual(row['document_id'],document_signature(good))
            self.assertEqual(row['row_group'],0)
            self.assertEqual(row['row_in_group'],0)
            self.assertFalse(row['inherited_holdout'])
            self.assertEqual(pq.read_table(source).column('content').to_pylist(),[good,good,held,reserved,bad])
            registry.write_text('[]')
            worker_setup([])


if __name__=='__main__':unittest.main()
