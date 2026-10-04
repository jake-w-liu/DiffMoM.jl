from pathlib import Path
import hashlib,shutil,tomllib
repo=Path(__file__).resolve().parent.parent
aspect_log=(repo/'data/plain_box_port_native_aspect_allowance_20261005.log').read_text(encoding='utf-8')
aspect=Path(aspect_log.split('Retained native snap audit: ')[-1].strip())
assert aspect.is_dir() and aspect.is_relative_to(repo/'data/planar_audit')
captures={'threshold':repo/'data/planar_audit/plain_box_port_native_boundary_allowance_1nyVbr',
          'tiny':repo/'data/planar_audit/plain_box_port_native_boundary_NdYvlt','aspect':aspect}
fixture=repo/'test/fixtures/native_box_port_boundary_allowance'
assert not fixture.exists()
reports={tag:tomllib.loads((folder/'comparison.toml').read_text(encoding='utf-8')) for tag,folder in captures.items()}
assert all(r['source_unchanged'] for r in reports.values())
assert [len(reports[k]['cases']) for k in captures]==[48,8,12]
fixture.mkdir()
def copy(source,target):
    destination=fixture/target
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
    assert source.read_bytes()==destination.read_bytes()
copy(Path(__file__),'assembly_producer.py')
rows=['scope = "Native port-edge allowance: both axes and ends, three native grids, MM/IN units, tiny excursions and rectangular unequal grids. Original strict candidate false rejections are retained unchanged."']
for group,capture in captures.items():
    copy(capture/'producer.jl',group+'/original_native_producer.jl')
    copy(capture/'comparison.toml',group+'/original_before_comparison.toml')
    for row in reports[group]['cases']:
        case=row['case'];name=group+'__'+case
        copy(capture/(case+'.son'),'cases/'+name+'/project.son')
        for filename in ('metadata.toml','engine_stdout.log','engine_stderr.log'):
            copy(capture/case/filename,'cases/'+name+'/native/'+filename)
        for filename in ('log_response.log','log_errors.log','log_composite.log','log_timing.log'):
            copy(capture/case/'sondata'/case/filename,'cases/'+name+'/native/'+filename)
        if row['native_status']=='ACCEPT':
            copy(capture/case/'native_raw.s2p','cases/'+name+'/native/native_raw.s2p')
        else:
            assert 'partially or entirely outside of the box' in row['native_stderr']
        rows.extend(['','[[cases]]',f'name = "{name}"',f'group = "{group}"',f'native_status = "{row["native_status"]}"'])
for filename in ('PlanarSonnetIO.jl','PlanarSonnetGeometryVariables.jl'):
    source=repo/'src/planar'/filename
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(r['source_before'][str(source.relative_to(repo))]==digest for r in reports.values())
    copy(source,'source_before/'+filename)
(fixture/'index.toml').write_text('\n'.join(rows)+'\n',encoding='utf-8')
files=sorted(p for p in fixture.rglob('*') if p.is_file())
(fixture/'sha256.toml').write_text('[sha256]\n'+''.join(f'"{p.relative_to(fixture).as_posix()}" = "{hashlib.sha256(p.read_bytes()).hexdigest()}"\n' for p in files),encoding='utf-8')
print(f'Sealed {len(files)+1} files, {sum(len(r["cases"]) for r in reports.values())} native controls')
