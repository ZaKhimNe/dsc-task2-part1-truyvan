"""B retrieval -> production QAPackage, usable directly or before experiments on Kaggle."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ChanTaooDe--main'))


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def strict_unit(doc, uid):
    """Avoid legacy get_unit_text's first-article fallback for missing gold IDs."""
    for article in doc['dieu']:
        if article['dieu_id'] == uid:
            return article['text']
        for clause in article['khoan']:
            if clause['khoan_id'] == uid:
                return clause['text']
    raise ValueError(f'Unknown unit: {uid}')


def context_item(doc, uid, score, expand):
    from src.b6_context_package.package import build_context_item
    cid = doc['context_id']
    text = strict_unit(doc, uid)
    covered = [uid]
    kind = 'doc_fallback' if doc['parse_status'] == 'fallback' else 'dieu_fallback'
    for article in doc['dieu']:
        if any(c['khoan_id'] == uid for c in article['khoan']):
            kind = 'khoan'
            if expand:
                text = article['text']
                covered = [article['dieu_id']] + [c['khoan_id'] for c in article['khoan']]
            break
        if article['dieu_id'] == uid:
            covered = [uid] + [c['khoan_id'] for c in article['khoan']]
    import re
    number = re.search(r'\b\d{1,4}/\d{4}/[\wĐđ-]+', doc.get('name', ''))
    item = build_context_item(cid, uid, kind, text, doc.get('name', ''), doc.get('link', ''),
                              number.group() if number else '', score)
    return {**item, 'context_id': cid, 'unit_id': uid, 'covered_unit_ids': covered}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--questions', required=True, help='QA dict or QAPackage list')
    p.add_argument('--corpus', required=True, help='Directory of context JSON files')
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--top-k', type=int, default=1)
    p.add_argument('--expand-article', action='store_true')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.top_k < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('top-k/limit must be positive')
    questions = json.loads(Path(args.questions).read_text(encoding='utf-8-sig'))
    if isinstance(questions, dict):
        rows = [{'id': str(q), 'question': r['question'], 'reference_answer': r.get('answer')}
                for q, r in questions.items()]
    else:
        rows = questions
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({str(r['id']) for r in rows}) != len(rows):
        raise ValueError('Questions must have unique IDs and be nonempty')
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    sources = sorted(Path(args.corpus).rglob('*.json'))
    if not sources:
        raise ValueError('Corpus has no JSON documents')
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(args.corpus)).encode())
        digest.update(source.read_bytes())
    fingerprint = {'corpus_sha256': digest.hexdigest(), 'encoder': 'BAAI/bge-m3', 'max_length': 1024,
                   'bridge_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'source_sha256': hashlib.sha256(b''.join(f.read_bytes() for f in
                       sorted((ROOT / 'ChanTaooDe--main' / 'src').rglob('*.py')))).hexdigest()}
    identity = cache / 'identity.json'
    if identity.exists() and json.loads(identity.read_text()) != fingerprint:
        raise ValueError('Cache belongs to different corpus/code: choose a new cache directory')
    if not identity.exists() and any(cache.iterdir()):
        raise ValueError('Nonempty cache without identity')
    write(identity, fingerprint)
    from src.b1_parser.parser import parse_document
    from src.common.io_utils import _normalize_context
    from src.b2_retrieval import retrieve
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder

    parsed_file = cache / 'parsed.json'
    if parsed_file.exists():
        parsed = json.loads(parsed_file.read_text(encoding='utf-8'))
    else:
        parsed = {}
        for source in sources:
            raw = _normalize_context(json.loads(source.read_text(encoding='utf-8-sig')))
            passage = raw.get('passage', '')
            if not passage.strip():
                continue
            cid = str(raw['id'])
            if cid in parsed:
                raise ValueError(f'Duplicate corpus ID: {cid}')
            doc = parse_document(cid, passage)
            parsed[cid] = {**doc, 'context_id': cid, 'name': raw.get('name', ''), 'link': raw.get('link', '')}
        write(parsed_file, parsed)
    index_file = cache / 'bm25.pkl'
    if index_file.exists():
        index = retrieve.load_index(index_file)
    else:
        raw_texts = {}
        for source in sources:
            raw = _normalize_context(json.loads(source.read_text(encoding='utf-8-sig')))
            if raw.get('passage', '').strip():
                raw_texts[str(raw['id'])] = raw['passage']
        index = retrieve.build_index(raw_texts)
        retrieve.save_index(index, index_file.with_suffix('.tmp'))
        index_file.with_suffix('.tmp').replace(index_file)
        del raw_texts
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer('BAAI/bge-m3', device=device)
    encoder.max_seq_length = 1024
    if device == 'cuda':
        encoder.half()
    emb_file = cache / 'embeddings.npy'
    unit_file = cache / 'unit_ids.json'
    if emb_file.exists() and unit_file.exists():
        embeddings = np.load(emb_file).astype(np.float32)
        unit_ids = json.loads(unit_file.read_text())
    else:
        units = {}
        for doc in parsed.values():
            for article in doc['dieu']:
                for unit in (article['khoan'] or [article]):
                    uid = unit.get('khoan_id', unit.get('dieu_id'))
                    units.setdefault(uid, unit['text'])  # match legacy first-duplicate behavior
        unit_ids = list(units)
        embeddings = encoder.encode(list(units.values()), batch_size=16, normalize_embeddings=True,
                                    show_progress_bar=True).astype(np.float32)
        with emb_file.with_suffix('.tmp').open('wb') as stream:
            np.save(stream, embeddings)
        emb_file.with_suffix('.tmp').replace(emb_file)
        write(unit_file, unit_ids)
        del units
    if len(unit_ids) != len(embeddings):
        raise ValueError('Embedding row/ID mismatch')
    query_embeddings = encoder.encode([r['question'] for r in rows], batch_size=16,
                                      normalize_embeddings=True).astype(np.float32)
    del encoder
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    emb_index = retrieve.build_corpus_embedding_index(unit_ids)
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device=device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    packages = []
    trace = output.with_suffix('.retrieval.jsonl')
    with trace.open('x', encoding='utf-8') as stream:
        for row, qemb in zip(rows, query_embeddings):
            candidates = retrieve.search_units_hybrid(index, row['question'], parsed, embeddings,
                                                       unit_ids, emb_index, qemb)
            scores = reranker.predict([(row['question'], u['text']) for u in candidates], batch_size=16) if candidates else []
            ranked = sorted([{**u, 'score': float(s)} for u, s in zip(candidates, scores)],
                            key=lambda u: u['score'], reverse=True)
            contexts = [context_item(parsed[str(u['context_id'])], u['unit_id'], u['score'], args.expand_article)
                        for u in ranked[:args.top_k]]
            packages.append({'id': str(row['id']), 'question': row['question'], 'contexts': contexts,
                             'reference_answer': row.get('reference_answer')})
            stream.write(json.dumps({'id': str(row['id']), 'ranked_units': ranked}, ensure_ascii=False) + '\n')
            stream.flush()
            print(f'Retrieval {len(packages)}/{len(rows)}', flush=True)
    write(output, packages)


if __name__ == '__main__':
    main()
