"""Prove extracted text remains attributable to its original training document."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from select_web_prose_spans import extract_file, merge
from test_web_document_quality import PROSE


class SpanExtractionTests(unittest.TestCase):
    def test_source_positions_and_exact_dedup_without_joining(self):
        raw = '标题\n  ' + PROSE + '  \n点击下方公众号领取资料。\n' + PROSE
        candidate = dict(file_index=0, row_group=1, row_in_group=0, candidate=True,
                         partition='train', inherited_holdout=False, document_id='fixture',
                         raw_sha256=hashlib.sha256(raw.encode()).hexdigest(), source='fixture', score=.8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.parquet'
            pq.write_table(pa.table({'content': ['not selected', raw]}), source, row_group_size=1)
            stats = extract_file((0, source, [candidate], root))
            self.assertEqual(stats['spans'], 2)
            spans = pq.read_table(root / 'spans-000.parquet').to_pylist()
            for span in spans:
                self.assertEqual(span['content'], raw[span['start']:span['end']])
                self.assertEqual(span['content'], PROSE)
            self.assertGreater(spans[1]['start'], spans[0]['end'])
            result = merge(root, [stats])
            self.assertEqual(result['totals']['unique_spans'], 1)
            self.assertEqual(result['totals']['duplicate_spans'], 1)
            self.assertEqual(result['totals']['unique_documents'], 1)
            with self.assertRaisesRegex(ValueError, 'training candidates'):
                extract_file((0, source, [dict(candidate, inherited_holdout=True)], root))
            with self.assertRaisesRegex(ValueError, 'changed since'):
                extract_file((0, source, [dict(candidate, raw_sha256='changed')], root))


if __name__ == '__main__':
    unittest.main()
