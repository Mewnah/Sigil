import json,sys,collections
def load(p): return json.load(open(p,encoding='utf-8'))
def main(paths, langs=None):
    rows=[];lrows=[];comps=[]
    for p in paths:
        d=load(p); rows+=d['rows']; lrows+=d['language_rows']; comps+=d.get('comparisons',[])
    conds=[]
    for r in rows:
        if r['condition'] not in conds: conds.append(r['condition'])
    byl=collections.defaultdict(dict)
    for r in rows: byl[(r['language'],r['label'])][r['condition']]=r
    for r in lrows: byl[(r['language'],r['label'])]['ALL']=r
    cols=conds+['ALL']
    print('| lang | engine | '+' | '.join(cols)+' | n |')
    print('|'+'---|'*(len(cols)+3))
    for (l,e) in sorted(byl):
        if langs and l not in langs: continue
        c=byl[(l,e)]
        cells=[]
        for k in cols:
            r=c.get(k)
            if r is None or r.get('rate') is None: cells.append('-'); continue
            lo,hi=(r.get('ci95') or [None,None])
            cells.append(f"{r['rate']*100:.1f}")
        n=c.get('ALL',{}).get('items','')
        print(f"| {l} ({next(iter(c.values()))['metric']}) | {e} | "+' | '.join(cells)+f" | {n} |")
    if comps:
        print('\npaired differences (a - b, WER points; negative = a better):')
        for c in comps:
            if langs and c['language'] not in langs: continue
            if c['condition']!='*': continue
            print(f"  {c['language']} {c['a_label']} vs {c['b_label']}: {c['delta']*100:+.1f} [{c['ci95'][0]*100:+.1f},{c['ci95'][1]*100:+.1f}] items={c['items']}")
if __name__=='__main__':
    main(sys.argv[1:])
