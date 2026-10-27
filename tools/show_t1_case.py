"""Print real saved T1 records for presentation screenshots; never calls an LLM."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def display(value, width=48):
    for paragraph in str(value).splitlines():
        print(textwrap.fill(paragraph, width=width, break_long_words=False, break_on_hyphens=False) if paragraph else '')

def summary():
    source = json.loads((ROOT/'metadata/sources.json').read_text(encoding='utf-8'))
    ready = [r for r in source if r['status']=='ready']
    bodies = [(ROOT/r['corpus_file']).read_text(encoding='utf-8').split('<!-- BODY -->',1)[1].strip() for r in ready]
    qa = rows(ROOT/'data/eval/qa_set.jsonl')
    docs = json.loads((ROOT/'index_langchain/documents.json').read_text(encoding='utf-8'))
    print('知识库与评测规模（读取项目实际文件）')
    print('入库文档:',len(ready),'  文本块:',len(docs))
    print('正文字符:',sum(len(t) for t in bodies),'  汉字:',sum(len(re.findall(r'[\u4e00-\u9fff]',t)) for t in bodies))
    print('PDF资料:',sum(bool(r.get('page_count')) for r in ready))
    print('评测题数:',len(qa),'  分层:',dict(Counter(q['level'] for q in qa)))
    print('\n已有T1结果检查（未重新调用模型）')
    for name in ['t1_baseline.jsonl','t1_no_retrieval.jsonl']:
        rs = rows(ROOT/'results'/name)
        print(name)
        print('  记录:',len(rs),'  成功调用:',sum(r['status']=='ok' for r in rs),'  截断:',sum(r.get('truncated',False) for r in rs))
    print('截图中的已有结果不是本次新推理。')

def index():
    sys.path.insert(0,str(ROOT/'src'))
    from kb import load_index, read_manifest, CHUNK_SIZE, CHUNK_OVERLAP
    kb=load_index()
    print('FAISS实际加载检查（未调用生成模型）')
    print('文档:',sum(r['status']=='ready' for r in read_manifest()))
    print('分块:',len(kb.documents),'  FAISS向量数:',kb.store.index.ntotal)
    print('向量维度:',kb.store.index.d)
    print('chunk_size:',CHUNK_SIZE,'  chunk_overlap:',CHUNK_OVERLAP)
    print('生成模型配置: qwen3:4b-instruct')
    print('检索默认top-k: 5')

def excerpt(row, brief):
    answer=row['answer']
    if not brief:
        return answer
    if row['condition']=='bare':
        if row['id']=='q003':
            return next(s for s in answer.splitlines() if '2门' in s)
        return answer[:180]+('…（摘录，全文在JSONL）' if len(answer)>180 else '')
    if row['id']=='q006':
        return '\n'.join(s.strip() for s in answer.splitlines() if any(needle in s for needle in ['累计认定不超过10','累计认定不超过15','后者比前者多']))
    if row['id']=='q001':
        return answer.splitlines()[0]
    if row['id']=='q003':
        return next(s for s in answer.splitlines() if '答案：' in s)
    if row['id']=='q017':
        return next(s for s in answer.splitlines() if '65931520' in s)
    if row['id']=='q019':
        return next(s for s in answer.splitlines() if '[8]' in s)
    return answer[:240]+('…（摘录，全文在JSONL）' if len(answer)>240 else '')

def case(qid, brief):
    rag=next((r for r in rows(ROOT/'results/t1_baseline.jsonl') if r['id']==qid),None)
    bare=next((r for r in rows(ROOT/'results/t1_no_retrieval.jsonl') if r['id']==qid),None)
    if rag is None or bare is None:
        raise SystemExit(f'找不到题号{qid}')
    assert rag['question']==bare['question']
    date=datetime.fromisoformat(rag['started_at']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    print(f'{qid}  {rag["level"]}  保存实验日期:{date}')
    print('模式：读取已有原始记录，未重新推理')
    display('问题：'+rag['question'])
    print('\n[实际top-5]')
    for r in rag['retrieved']:
        print(f'{r["rank"]}. {r["chunk_id"]}')
    print('Gold:',', '.join(rag['gold_chunk_ids']))
    print('命中:',len(rag.get('gold_hits',[])),'/',len(rag['gold_chunk_ids']))
    print('\n[RAG原始答案'+('摘录' if brief else '')+']')
    display(excerpt(rag,brief))
    print('\n[无检索原始答案'+('摘录' if brief else '')+']')
    display(excerpt(bare,brief))
    if not rag.get('citation_ids_valid',True):
        print('\n引用检查：存在越界编号，不能当作有效引用。')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('id',nargs='?',help='qa_set.jsonl中的题号，如q002')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--summary',action='store_true')
    mode.add_argument('--index',action='store_true')
    parser.add_argument('--brief',action='store_true',help='只显示标注为摘录的原文，方便截屏')
    args=parser.parse_args()
    if args.summary:
        summary()
    elif args.index:
        index()
    elif args.id:
        case(args.id,args.brief)
    else:
        parser.error('请提供题号，或使用--summary/--index')

if __name__=='__main__':
    main()
