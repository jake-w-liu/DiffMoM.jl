from pathlib import Path
import tempfile,subprocess,json,hashlib
root=Path.cwd();out=Path(tempfile.mkdtemp(prefix='radial_three_pass_whole_port_',dir=root/'data/planar_audit'))
source=root/'data/planar_audit/radial_three_pass_whole_selector_E2ZZFC/repeated_whole_parameter.son'
current=out/'parameter.son';current.write_bytes(source.read_bytes());stages=[]
for n in range(1,4):
 dest=out/f'stage{n}.son';command=['C:/Program Files/Sonnet Software/18.53/bin/soncmd.exe','-ReadWrite',str(current),str(dest)];r=subprocess.run(command,capture_output=True,text=True);(out/f'stage{n}.log').write_text(r.stdout+r.stderr,encoding='utf-8');assert r.returncode==0
 lines=dest.read_text(encoding='utf-8').splitlines();i=lines.index('NUM 3');j=i+1;polys=[]
 for _ in range(3):
  count=int(lines[j].split()[1]);polys.append(lines[j+1:j+1+count]);j+=count+2
 row={'stage':n,'polygons':polys,'nom':[l for l in lines if l.startswith('NOM ')]};stages.append(row);print(row);current=dest
(out/'observations.json').write_text(json.dumps({'stages':stages,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8');print(out)
