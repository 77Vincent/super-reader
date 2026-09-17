const test = require("node:test");
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { mkdtempSync, rmSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");

test("full Wikipedia supports uncapped documents and sequence lengths", async () => {
  const { parseArguments, cappedDocumentSamples } = await import("../training/prepare_full_wikipedia_data.mjs");
  const options = parseArguments(["--max-samples-per-document", "0", "--max-sequence-length", "0"]);
  assert.equal(options.maxSamplesPerDocument, 0);
  assert.equal(options.maxSequenceLength, 0);
  const candidates = Array.from({ length: 300 }, (_, index) => index);
  assert.deepEqual(cappedDocumentSamples(candidates, 0), candidates);
  assert.equal(cappedDocumentSamples(candidates, 128).length, 128);
  assert.throws(() => parseArguments(["--max-samples-per-document", "-1"]));
});

test("all local CLUE entries retain original IDs and give extra splits distinct IDs", async () => {
  const { localClueDocuments, documentFingerprint } = await import("../training/prepare_retraining_base.mjs");
  const directory = mkdtempSync(join(tmpdir(), "reader-local-clue-"));
  const archive = join(directory, "news.zip");
  try {
    execFileSync("python3", ["-c", `
import json,sys,zipfile
with zipfile.ZipFile(sys.argv[1], 'w') as archive:
 for entry in ['train.json','dev.json','test.json','test1.0.json']:
  archive.writestr(entry, json.dumps({'sentence':'苹果、香蕉，准备做果汁。'},ensure_ascii=False)+'\\n')
`, archive]);
    const documents = await localClueDocuments({ domain: "news", entries: ["train.json"] }, { primary: { path: archive } });
    assert.equal(documents.length, 4);
    assert.equal(documents[0].id, "news:0");
    assert.equal(new Set(documents.map(({ id }) => id)).size, 4);
    assert.equal(documentFingerprint(documents[0].text), documentFingerprint("苹果香蕉；准备做果汁！"));
    assert.notEqual(documentFingerprint(documents[0].text), documentFingerprint("完全不同的文章。"));
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("synthetic preparation exhausts cached files, removes holdout copies and retains bounded mode", () => {
  execFileSync("python3", ["-c", String.raw`
import sys,json,tempfile,hashlib
from pathlib import Path
root=Path.cwd()
sys.path[:0]=[str(root/'training/.deps'),str(root/'training')]
from run_prepare_synthetic import ensure_dependencies
ensure_dependencies()
import pyarrow as pa
import pyarrow.parquet as pq
import prepare_synthetic_data as p
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory).resolve()
 p.PROJECT_DIR=root
 p.BLOOM_BYTES=1024*1024
 p.BLOOM_BIT_MASK=p.BLOOM_BYTES*8-1
 raw=root/'raw'; raw.mkdir()
 data=root/'data'; data.mkdir()
 for split in ['validation','test']:
  (data/(split+'.jsonl')).write_text('')
 heldout='引句。'+'保护留出文章'*20+'，'+'不能用于训练'*20+'。'
 (data/'holdout-document-hashes.json').write_text(json.dumps([p.document_signature(heldout)]))
 downloads=[]
 rows=[['引句。'+'甲'*90+'，'+'乙'*90+'。',heldout],['引句。'+'丙'*90+'，'+'丁'*90+'。','引句。'+'戊'*90+'，'+'己'*90+'。']]
 for index,values in enumerate(rows):
  path=raw/f'{index:05d}.parquet'
  pq.write_table(pa.table({'content':values}),path)
  downloads.append({'source_index':index,'source_url':f'https://example.invalid/{index}',
                    'sha256':p.sha256_file(path)})
 source=root/'source.json'
 source.write_text(json.dumps({'synthetic_source':{'dataset':p.DATASET,'config':p.CONFIG,'split':p.SPLIT,'downloads':downloads}}))
 base=root/'base.json'
 stats=p.empty_statistics(10); stats['domain_samples']={'news':0}
 base.write_text(json.dumps({**p.DATA_POLICY,'domains':['news'],'shards':[],
                            'statistics':stats,'evaluation_source_dir':str(data),'vocabulary_path':'vocabulary.json'}))
 p.urllib.request.urlopen=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('Unexpected network access'))
 for target,expected_count,expected_rows in [(0,3,4),(1,1,1)]:
  output=root/f'out-{target}'
  sys.argv=['prepare','--base-manifest',str(base),'--output-dir',str(output),'--raw-dir',str(raw),
            '--source-manifest',str(source),'--target-samples',str(target),'--synthetic-shards','2',
            '--max-samples-per-document','0','--max-sequence-length','0']
  p.main()
  result=p.load_json(output/'manifest.json')
  state=p.load_json(output/'preparation-state.json')
  assert result['statistics']['samples']==expected_count
  assert state['statistics']['documents_seen']==expected_rows
  assert result['source_exhausted']==(target==0)
  assert result['statistics']['overlength_filtered']==0
  assert result['statistics']['document_sample_cap_filtered']==0
  if target==0:
   assert result['statistics']['holdout_documents_filtered']==1
   assert state['source_file_index']==2
   samples=[item for shard in result['shards'] for item in p.iter_json_lines(p.project_path(shard['path']))]
   assert len(samples)==3 and all('保护' not in item[0] for item in samples)
 assert p.evenly_capped(list(range(300)),0)==list(range(300))
`], { stdio: "pipe", timeout: 180000 });
});
