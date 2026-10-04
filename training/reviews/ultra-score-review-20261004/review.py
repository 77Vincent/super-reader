"""Append explicit AI reading decisions; no automatic semantic annotation."""
from pathlib import Path
import argparse
import json

HERE=Path(__file__).resolve().parent
CODES={
 'A':('A','ordinary','标记处的分句、句间、并列、话题或说明边界可接受；不要求唯一切点。'),
 'B':('A','boilerplate','边界可接受，但内容是推广、版权声明、订阅或网页操作等附属文字。'),
 'D':('A','damaged','边界尚可接受，但文字存在明显插入、缺失或语义表达损坏。'),
 'U':('U','uncertain','当前片段不足以确定边界或文字完整性，需查看原文。'),
 'X':('U','damaged','文字损坏使边界本身也无法可靠判断。'),
 'E':('E','defective','有可定位的错误切点，需逐条说明。'),
}


def record(start,codes,notes=None):
 rows=json.loads((HERE/'sample.json').read_text())
 path=HERE/'annotations.json'
 previous=json.loads(path.read_text()) if path.exists() else []
 assert not (HERE/'freeze.json').exists()
 assert len(previous)==start-1
 codes=codes.split();assert len(codes)==80
 notes=notes or {}
 for i,code in enumerate(codes,start):
  boundary,text_type,reason=CODES[code]
  note=notes.get(str(i))
  if code in ['D','X','E','U']:assert note,i
  previous.append(dict(review_id=rows[i-1]['review_id'],code=code,boundary=boundary,
                       text_type=text_type,reason=note or reason,reviewer='Codex AI first pass'))
 path.write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n')
 print('Saved',start,'to',len(previous))


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('start',type=int);args=parser.parse_args()
 rows=json.loads((HERE/'sample.json').read_text())
 for r in rows[args.start-1:args.start+79]:
  text=r['text'];cut=r['cut'];print(r['review_id'],text[:cut]+'【切】'+text[cut:])
