from pathlib import Path
import tomllib,hashlib,shutil
repo=Path(__file__).resolve().parent.parent
capture=repo/'data/planar_audit/moved_box_port_native_allowance_t05UDl'
fixture=repo/'test/fixtures/native_moved_box_port_boundary_allowance'
report=tomllib.loads((capture/'comparison.toml').read_text(encoding='utf-8'))
assert report['source_unchanged'] and len(report['cases'])==3 and not fixture.exists()
fixture.mkdir()
def copy(source,target):
    target=fixture/target;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    assert source.read_bytes()==target.read_bytes()
copy(Path(__file__),'assembly_producer.py')
copy(capture/'producer.jl','original_native_producer.jl')
copy(capture/'comparison.toml','comparison.toml')
lines=['scope = "Three active ANC parameter/literal pairs around the native port-edge allowance. Accepted and explicit rejected inputs, logs and geometry outcomes are retained unchanged."']
for row in report['cases']:
    name=row['case'];status=row['parameter_native_status']
    assert status==row['literal_native_status']==row['geometry_status']
    for suffix in ('parameter','literal'):
        copy(capture/(name+'_'+suffix+'.son'),'cases/'+name+'/'+suffix+'.son')
        for filename in ('metadata.toml','engine_stdout.log','engine_stderr.log'):
            copy(capture/name/suffix/filename,'cases/'+name+'/'+suffix+'/'+filename)
        for filename in ('log_response.log','log_errors.log','log_composite.log','log_timing.log'):
            copy(capture/name/suffix/'sondata'/(name+'_'+suffix)/filename,'cases/'+name+'/'+suffix+'/'+filename)
        if status=='ACCEPT':
            copy(capture/name/suffix/'native_raw.s2p','cases/'+name+'/'+suffix+'/native_raw.s2p')
        else:
            assert 'partially or entirely outside of the box' in row[suffix+'_native_stderr']
    lines.extend(['','[[cases]]',f'name = "{name}"',f'native_status = "{status}"'])
(fixture/'index.toml').write_text('\n'.join(lines)+'\n',encoding='utf-8')
files=sorted(p for p in fixture.rglob('*') if p.is_file())
(fixture/'sha256.toml').write_text('[sha256]\n'+''.join(f'"{p.relative_to(fixture).as_posix()}" = "{hashlib.sha256(p.read_bytes()).hexdigest()}"\n' for p in files),encoding='utf-8')
print('Sealed moved allowance fixture',len(files)+1,'files')
