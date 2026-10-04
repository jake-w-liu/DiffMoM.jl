from pathlib import Path
import hashlib, shutil, subprocess, tomllib
repo=Path(__file__).resolve().parent.parent
capture=repo/'data/planar_audit/plain_box_port_extent_69slJp'
conformal=repo/'data/planar_audit/plain_box_port_extent_conformal_O3wMWK'
fixture=repo/'test/fixtures/native_plain_box_port_extent'
assert not fixture.exists(), 'refuse to replace sealed evidence'
report=tomllib.loads((capture/'comparison.toml').read_text(encoding='utf-8'))
assert report['source_unchanged'] and len(report['cases'])==6
fixture.mkdir()
def copy(source,target):
    destination=fixture/target
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
    assert source.read_bytes()==destination.read_bytes()
copy(Path(__file__),'assembly_producer.py')
copy(capture/'producer.jl','original_native_producer.jl')
copy(capture/'comparison.toml','original_before_comparison.toml')
copy(conformal/'producer.jl','conformal_followup_producer.jl')
copy(conformal/'comparison.toml','conformal_followup_comparison.toml')
for row in report['cases']:
    tag=row['case']
    copy(capture/(tag+'.son'),'cases/'+tag+'/project.son')
    for name in ('metadata.toml','engine_stdout.log','engine_stderr.log'):
        copy(capture/tag/name,'cases/'+tag+'/native/'+name)
    for name in ('log_response.log','log_errors.log','log_composite.log','log_timing.log'):
        copy(capture/tag/'sondata'/tag/name,'cases/'+tag+'/native/'+name)
    if row['native_status']=='ACCEPT':
        copy(capture/tag/'native_raw.s2p','cases/'+tag+'/native/native_raw.s2p')
        copy(capture/tag/'sondata'/tag/'jxy/current_1.sid','cases/'+tag+'/native/current_1.sid')
    else:
        assert 'partially or entirely outside of the box' in row['native_stderr']
    assert row['raster_status']=='ACCEPT'
old=subprocess.run(['git','show','HEAD:src/planar/PlanarSonnetIO.jl'],cwd=repo,check=True,stdout=subprocess.PIPE).stdout
expected=report['source_before'][r'src\planar\PlanarSonnetIO.jl']
assert hashlib.sha256(old).hexdigest()==expected
(fixture/'source_before').mkdir()
(fixture/'source_before/PlanarSonnetIO.jl').write_bytes(old)
rows=['scope = "Six independent native literal box-port edge controls; original raster false acceptances and corrected conformal follow-up are retained unchanged."',
      'source_before_commit = "9f3324915702f15784f9a34266a180548b63eafc"']
for row in report['cases']:
    rows.extend(['','[[cases]]',f'name = "{row["case"]}"',f'native_status = "{row["native_status"]}"',f'lo = {row["lo"]}',f'hi = {row["hi"]}'])
(fixture/'index.toml').write_text('\n'.join(rows)+'\n',encoding='utf-8')
files=sorted(p for p in fixture.rglob('*') if p.is_file())
(fixture/'sha256.toml').write_text('[sha256]\n'+''.join(f'"{p.relative_to(fixture).as_posix()}" = "{hashlib.sha256(p.read_bytes()).hexdigest()}"\n' for p in files),encoding='utf-8')
print(f'Sealed {len(files)+1} fixture files at {fixture}')
