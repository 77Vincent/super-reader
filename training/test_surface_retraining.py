"""Small offline checks for preparation rollback and target-independent holdout protection."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
import prepare_synthetic_data as prep
from filter_retraining_holdouts import projected_input_signature
from text_policy import DATA_POLICY, training_pairs

class SurfaceRetrainingTests(unittest.TestCase):
    def test_input_identity_ignores_old_gap_and_punctuation(self):
        self.assertEqual(projected_input_signature('甲:8.30乙丙',1),projected_input_signature('甲乙、丙',0))
        self.assertNotEqual(projected_input_signature('甲乙丙'),projected_input_signature('甲乙丁'))

    def test_fillers_and_emoji_are_distinct_from_plain_spaces(self):
        self.assertEqual(list(training_pairs('b的钢材，表示屈服点为_ MPa的A级。')),[])
        self.assertEqual(len(list(training_pairs('调用__init__函数，随后继续。'))),1)
        self.assertEqual(len(list(training_pairs('普通  空格保留，👨‍👩‍👧‍👦一家出游。'))),1)

    def fixture(self,root):
        raw=root/'raw';raw.mkdir();base=root/'base';base.mkdir()
        for split in ['validation','test']:(base/(split+'.jsonl')).write_text('')
        (base/'holdout-document-hashes.json').write_text('[]')
        manifest={**DATA_POLICY,'domains':['news'],'shards':[],'statistics':prep.empty_statistics(10),
            'evaluation_source_dir':str(base),'vocabulary_path':str(base/'vocabulary.json')}
        manifest['statistics']['domain_samples']={'news':0}
        (base/'manifest.json').write_text(json.dumps(manifest))
        texts=['\n'.join('，'.join(chr(0x6000+i)+chr(0x6100+j)+('自然中文正文内容'*4) for j in range(4))+'。' for _ in range(2)) for i in range(3)]
        parquet=raw/'00000.parquet';pq.write_table(pa.table({'content':texts}),parquet)
        source=root/'source.json';source.write_text(json.dumps({'synthetic_source':{'dataset':prep.DATASET,'config':prep.CONFIG,'split':prep.SPLIT,
            'downloads':[{'source_index':0,'source_url':'https://unused.invalid/fixture.parquet','sha256':prep.sha256_file(parquet)}]}}))
        return argparse.Namespace(base_manifest=base/'manifest.json',output_dir=root/'complete',raw_dir=raw,source_manifest=source,
            target_samples=0,synthetic_shards=2,max_sequence_length=0,max_samples_per_document=0,batch_rows=2,checkpoint_documents=1)

    def invoke(self,args):
        with patch.object(prep,'parse_arguments',return_value=args),patch.object(prep,'PROJECT_DIR',args.output_dir.parent.resolve()),patch.object(prep,'BLOOM_BYTES',1024*1024),patch.object(prep,'BLOOM_BIT_MASK',8*1024*1024-1),contextlib.redirect_stdout(io.StringIO()):
            prep.main()

    def test_resume_truncates_uncommitted_tail_before_rebuilding_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);self.invoke(args)
            expected=[p.read_bytes() for p in sorted(args.output_dir.glob('synthetic-train-*.jsonl'))]
            resumed=argparse.Namespace(**{**vars(args),'output_dir':root/'resumed'})
            real_write=prep.write_json_atomic
            def interrupt(path,value):
                real_write(path,value)
                if path.name=='preparation-state.json' and value['statistics']['documents_seen']==1:
                    raise InterruptedError('fixture stop')
            with patch.object(prep,'write_json_atomic',side_effect=interrupt):
                with self.assertRaises(InterruptedError):self.invoke(resumed)
            with (resumed.output_dir/'synthetic-train-000.jsonl.part').open('ab') as handle:
                handle.write(b'UNCOMMITTED INVALID JSON')
            self.invoke(resumed)
            actual=[p.read_bytes() for p in sorted(resumed.output_dir.glob('synthetic-train-*.jsonl'))]
            self.assertEqual(actual,expected)
            for final in resumed.output_dir.glob('synthetic-train-*.jsonl'):
                self.assertEqual(final.stat().st_ino,final.with_name(final.name+'.part').stat().st_ino)
            self.invoke(resumed)

if __name__=='__main__':unittest.main()
