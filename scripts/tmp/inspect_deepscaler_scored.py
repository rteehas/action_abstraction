from datasets import load_from_disk
from pathlib import Path
p = Path('/deepscaler_qwne1_7B_solutions_scored')
ds = load_from_disk(str(p))
print(type(ds).__name__)
if hasattr(ds, 'keys'):
    print('splits', list(ds.keys()))
    for k,v in ds.items():
        print(k, len(v))
        print('features', list(v.features.keys())[:40])
        print('sample keys', list(v[0].keys())[:40])
        print('sample passrate', v[0].get('passrate'))
        break
else:
    print('length', len(ds))
    print('features', list(ds.features.keys())[:40])
    print('sample keys', list(ds[0].keys())[:40])
    print('sample passrate', ds[0].get('passrate'))
