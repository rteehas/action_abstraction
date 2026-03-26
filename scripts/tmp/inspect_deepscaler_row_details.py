from datasets import load_from_disk
from pathlib import Path
from pprint import pprint
p = Path('/deepscaler_qwne1_7B_solutions_scored')
ds = load_from_disk(str(p))
for idx in [0, 1, 10, 100]:
    row = ds[idx]
    print('ROW', idx)
    print('answer', row['answer'])
    print('num_correct', row.get('num_correct'))
    print('passrate', row.get('passrate'))
    print('solution_type', type(row.get('solution')).__name__)
    print('generated_solution_type', type(row.get('generated_solution')).__name__)
    print('generated_answer_type', type(row.get('generated_answer')).__name__)
    if isinstance(row.get('generated_solution'), list):
        print('num_generated_solution', len(row['generated_solution']))
        print('first_solution_prefix', (row['generated_solution'][0] or '')[:300])
    if isinstance(row.get('generated_answer'), list):
        print('num_generated_answer', len(row['generated_answer']))
        print('generated_answers', row['generated_answer'][:10])
    print('solution_prefix', (row.get('solution') or '')[:300])
    print('keys', list(row.keys()))
    print('---')
