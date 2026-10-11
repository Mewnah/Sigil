import json,sys,os
KEYS=['items','errors','compute_ms_p50','compute_ms_p95','rtf_mean','rtf_p95','final_latency_ms_p50','final_latency_ms_p95','first_partial_ms_p50','first_partial_ms_p95','latency_ms_p50','latency_ms_p95','ttfa_ms_p50','ttfa_ms_p95','cpu_cores_mean','cpu_ms_per_audio_s','ram_peak_mb','vram_peak_mb','chrf']
def fmt(v):
    if v is None: return '-'
    if isinstance(v,float): return f'{v:.3g}' if abs(v)<10 else f'{v:.0f}'
    return str(v)
def main(paths, keys=None):
    keys=keys or KEYS
    for p in paths:
        r=json.load(open(p,encoding='utf-8'))
        print('##',os.path.basename(os.path.dirname(p)), r.get('spec',{}).get('name') if isinstance(r.get('spec'),dict) else '')
        for run in r['runs']:
            print('run',run['name'],run['mode'],run['status'],(run.get('error') or '')[:100])
            ks=[k for k in keys if any(e['aggregate'].get(k) is not None for e in run.get('results',[]))]
            print('| engine | '+' | '.join(ks)+' |')
            for e in run.get('results',[]):
                a=e['aggregate']
                print('| '+e['engine']+' | '+' | '.join(fmt(a.get(k)) for k in ks)+' |')
            for w in run.get('warnings',[])[:6]: print('  warn:',w[:160])
if __name__=='__main__':
    main(sys.argv[1:])
