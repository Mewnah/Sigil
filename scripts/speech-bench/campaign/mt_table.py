import json,collections,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
acc=collections.defaultdict(lambda:[0.0,0])
for r in d['rows']:
    s,eng=r['label'].split('/')
    key=(r['direction'],eng)
    acc[key][0]+=r['chrf']*r['lines']; acc[key][1]+=r['lines']
engs=sorted({k[1] for k in acc})
dirs=sorted({k[0] for k in acc})
print('| direction | '+' | '.join(engs)+' |')
print('|---|'+'---|'*len(engs))
tot=collections.defaultdict(lambda:[0.0,0])
for dr in dirs:
    cells=[]
    for e in engs:
        v=acc.get((dr,e))
        cells.append(f'{v[0]/v[1]:.1f}' if v else '-')
        if v: tot[e][0]+=v[0]; tot[e][1]+=v[1]
    print(f'| {dr} | '+' | '.join(cells)+' |')
