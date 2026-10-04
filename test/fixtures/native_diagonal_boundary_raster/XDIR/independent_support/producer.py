from pathlib import Path
import hashlib,json,shutil,tempfile
from collections import deque

repo=Path(__file__).resolve().parents[1]
native=repo/'data/sonnet_validation/zero_nominal_literal_mesh_31cDAe'
public=repo/'data/planar_audit/zero_nominal_literal_public_masks_TbNIzn'
output=Path(tempfile.mkdtemp(prefix='zero_nominal_literal_mesh_comparison_',dir=repo/'data/planar_audit'))
shutil.copyfile(__file__,output/'producer.py')
paths=[Path(__file__),native/'comparison.toml',public/'comparison.toml']+list((repo/'src').rglob('*.jl'))
for target in (.0625,.125):
    tag=f'XDIR_dir_-1_target_{target}_literal'
    paths.extend([native/tag/'sondata'/tag/'jxy/current_1.sid',public/f'{tag}_mask.txt'])
def hashes():
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
def components(cells):
    remaining=set(cells);sizes=[]
    while remaining:
        queue=deque([remaining.pop()]);size=0
        while queue:
            x,y=queue.popleft();size+=1
            for other in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if other in remaining:
                    remaining.remove(other);queue.append(other)
        sizes.append(size)
    return sorted(sizes,reverse=True)
report={'scope':'Independent comparison of actual public cell masks and union of native axis-aligned X/Y subsection rectangles, clipped to the physical box. This support union is an observable, not a claim that every isolated metal cell is represented by a current basis. No acceptance gates change.','source_before':hashes(),'cases':[]}
for target in (.0625,.125):
    tag=f'XDIR_dir_-1_target_{target}_literal'
    mask=(public/f'{tag}_mask.txt').read_text().splitlines()
    nx,ny=len(mask[0]),len(mask)
    assert all(len(row)==nx and set(row)<={'0','1'} for row in mask)
    actual={(x,y) for y,row in enumerate(mask) for x,value in enumerate(row) if value=='1'}
    rows=(native/tag/'sondata'/tag/'jxy/current_1.sid').read_text().splitlines()
    start=next(i for i,row in enumerate(rows) if row.startswith('SUB '))
    count=int(rows[start].split()[2]);rectangles=rows[start+1:start+1+count]
    assert rows[start+1+count]=='END'
    supports={'11':set(),'12':set()}
    for rectangle in rectangles:
        fields=rectangle.split();assert len(fields)==8 and fields[1] in supports
        xmin,ymin,xmax,ymax=map(int,fields[2:6])
        assert xmin<xmax and ymin<ymax and all(value%2==0 for value in (xmin,ymin,xmax,ymax))
        supports[fields[1]].update((x,y) for y in range(max(0,ymin//2),min(ny,ymax//2))
            for x in range(max(0,xmin//2),min(nx,xmax//2)))
    union=supports['11']|supports['12']
    row={'case':tag,'grid':[nx,ny],'subsections':count,'public_cells':len(actual),'native_support_cells':len(union),
        'public_components':components(actual),'native_support_components':components(union),
        'public_only_cells':sorted(actual-union),'native_only_cells':sorted(union-actual),'support_equal':actual==union}
    report['cases'].append(row);print(json.dumps(row))
report['source_after']=hashes();report['source_unchanged']=report['source_before']==report['source_after']
assert report['source_unchanged']
(output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('Retained mesh comparison:',output)
