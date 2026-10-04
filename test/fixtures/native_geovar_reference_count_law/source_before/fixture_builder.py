from pathlib import Path
import hashlib, json, shutil, tomllib

repo = Path(__file__).resolve().parents[1]
fixture = repo / 'test/fixtures/native_geovar_reference_count_law'
assert not fixture.exists()
fixture.mkdir()
cases = []
captures = {
    'count34': 'geovar_reference_count_law_probe_2xgDYY',
    'extended': 'geovar_reference_count_law_extended_CttacU',
    'orientation': 'geovar_reference_orientation_DrCyGL',
    'direction': 'geovar_reference_orientation_resolved_3drdml',
    'aligned_negative': 'geovar_reference_count_direction_I7ln87',
    'whole': 'geovar_reference_repeated_whole_tARe4w',
}
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = repo / 'data/planar_audit/geovar_reference_literal_baseline_rpFclH/geometry_before.jl'
source_before = fixture / 'source_before'
source_before.mkdir()
shutil.copy2(before, source_before / 'PlanarSonnetGeometryVariables.jl')
shutil.copy2(repo / 'validation/sonnet_stripline/sonnet_reference.jl', source_before / 'sonnet_reference.jl')

def copy_run(source, target):
    target.mkdir(parents=True)
    for name in ['metadata.toml', 'engine_stdout.log', 'engine_stderr.log', 'native_raw.s2p']:
        shutil.copy2(source / name, target / name)
    projects = list(source.glob('*.son'))
    assert len(projects) == 1
    shutil.copy2(projects[0], target / projects[0].name)
    meta = tomllib.loads((source / 'metadata.toml').read_text(encoding='utf-8'))
    assert meta['process_success'] and not meta['deembedded']
    assert meta['source_sha256'] == digest(projects[0])
    assert meta['touchstone_selected_log_checks']['native_raw.s2p']['status'] == 'PASS'
    sid = list((source / 'sondata').rglob('current_1.sid'))
    assert len(sid) <= 1
    if sid:
        shutil.copy2(sid[0], target / 'current_1.sid')

def add_case(category, tag, capture, parameter_stem, literal_stem, multiplicity, baseline=None):
    name = category + '__' + tag
    folder = fixture / 'cases' / name
    folder.mkdir(parents=True)
    for stem, suffix in [(parameter_stem, 'parameter'), (literal_stem, 'literal')]:
        source = capture / (stem + '.son')
        shutil.copy2(source, folder / (suffix + '.son'))
        copy_run(capture / stem, folder / suffix)
        meta = tomllib.loads((folder / suffix / 'metadata.toml').read_text(encoding='utf-8'))
        assert meta['source_sha256'] == digest(folder / (suffix + '.son'))
    row = dict(name=name, category=category, explicit_reference_count=multiplicity,
               original_capture=capture.relative_to(repo).as_posix(),
               parameter_sha256=digest(folder / 'parameter.son'), literal_sha256=digest(folder / 'literal.son'))
    if baseline:
        row.update(baseline)
    cases.append(row)

baseline = tomllib.loads((repo / 'data/planar_audit/geovar_reference_literal_baseline_rpFclH/comparison.toml').read_text(encoding='utf-8'))
direction_baseline = tomllib.loads((repo / 'data/planar_audit/geovar_reference_direction_baseline_xg6OwO/comparison.toml').read_text(encoding='utf-8'))
for proof in [baseline, direction_baseline]:
    assert proof['source_unchanged'] and all(x['status'] == 'PASS' for x in proof['cases'])
observations = {x['case']: x for x in baseline['cases'] + direction_baseline['cases']}
for category, directory in captures.items():
    capture = repo / 'data/planar_audit' / directory
    proof = tomllib.loads((capture / 'comparison.toml').read_text(encoding='utf-8'))
    assert proof['source_unchanged'] and proof['source_before'] == proof['source_after']
    assert {k.replace('\\','/'): v for k,v in proof['source_before'].items()}['src/planar/PlanarSonnetGeometryVariables.jl'] == digest(before)
    provenance = fixture / 'provenance' / category
    provenance.mkdir(parents=True)
    for name in ['comparison.toml', 'producer.jl']:
        shutil.copy2(capture / name, provenance / name)
    for row in proof['cases']:
        tag = row['case']
        if category == 'count34':
            r = row['explicit_reference_count']; factor = 1 + r*(r-1)
            assert factor in row['matching_factors']
            literal_stem = tag + '_factor_' + str(factor)
        elif category == 'extended':
            if row['status'] != 'PASS':
                rejected = fixture / 'rejected_hypotheses' / tag
                for suffix in ['parameter', 'literal']:
                    shutil.copytree(capture / (tag + '_' + suffix), rejected / suffix,
                                    ignore=shutil.ignore_patterns('sondata'))
                    shutil.copy2(capture / (tag + '_' + suffix + '.son'), rejected / (suffix + '.son'))
                continue
            r = row['explicit_reference_count']; literal_stem = tag + '_literal'
        elif category == 'orientation':
            r = row['r']
            if r < 2:
                continue  # These small movements were deliberately ambiguous at the native grid.
            factor = (1+r*(r-1)) * (1 if tag.startswith('zero_') else -1)
            assert factor in row['matching_factors']
            literal_stem = tag + '_factor' + str(factor)
        elif category == 'direction':
            r = row['r']; assert row['matching_factors'] == [-1]
            literal_stem = tag + '_factor-1'
        elif category == 'aligned_negative':
            if row['direction'] != -1 or row['orientation'] != -1:
                continue
            r = row['r']; factor = 1+r*(r-1)
            assert row['matching_factors'] == [factor]
            literal_stem = tag + '_factor' + str(factor)
        else:
            r = row['r']; assert row['status'] == 'PASS'; literal_stem = tag + '_literal'
        old = observations.get(tag)
        observation = None if old is None else {k: old[k] for k in
            ['full_s_error', 'original_voltage_residual', 'before_parameter_full_s_error'] if k in old}
        add_case(category, tag, capture, tag + '_parameter', literal_stem, r, observation)

factor3 = repo / 'data/planar_audit/geovar_multiple_reference_factor3_probe_XqABap'
original = repo / 'test/fixtures/native_geovar_reference_variants/native'
for tag in ['nscd_expand_2', 'rad_expand_2', 'rad_contract_2']:
    folder = fixture / 'cases' / ('factor3__' + tag)
    folder.mkdir(parents=True)
    for source, suffix in [(original, 'parameter'), (factor3, 'literal')]:
        stem = tag + '_' + suffix
        shutil.copy2(source / (stem + '.son'), folder / (suffix + '.son'))
        copy_run(source / stem, folder / suffix)
    cases.append(dict(name='factor3__'+tag, category='factor3', explicit_reference_count=2,
                      parameter_sha256=digest(folder / 'parameter.son'), literal_sha256=digest(folder / 'literal.son'),
                      full_s_error=observations[tag]['full_s_error'], original_voltage_residual=observations[tag]['original_voltage_residual']))
assert len(cases) == 94
shutil.copy2(repo / 'data/geovar_reference_literal_physical_baseline_20261005.jl', source_before / 'literal_physical_producer.jl')
shutil.copy2(repo / 'data/planar_audit/geovar_reference_literal_baseline_rpFclH/comparison.toml', source_before / 'literal_physical.toml')
shutil.copy2(repo / 'data/geovar_reference_direction_physical_baseline_20261005.jl', source_before / 'direction_physical_producer.jl')
shutil.copy2(repo / 'data/planar_audit/geovar_reference_direction_baseline_xg6OwO/comparison.toml', source_before / 'direction_physical.toml')
shutil.copy2(Path(__file__), source_before / 'fixture_builder.py')
quote = lambda x: json.dumps(x, ensure_ascii=False)
with (fixture / 'index.toml').open('w', encoding='utf-8', newline='\n') as f:
    f.write('scope = "Native ANC NSCD reference-coordinate direction and ANC/RAD repeated-reference law; all historical captures immutable"\n')
    for row in cases:
        f.write('\n[[cases]]\n')
        for key, value in row.items():
            f.write(key + ' = ' + (quote(value) if isinstance(value,str) else repr(value)) + '\n')
files = sorted(p for p in fixture.rglob('*') if p.is_file())
with (fixture / 'sha256.toml').open('w', encoding='utf-8', newline='\n') as f:
    f.write('[sha256]\n')
    for p in files:
        f.write(quote(p.relative_to(fixture).as_posix()) + ' = ' + quote(digest(p)) + '\n')
print('Retained94 native pairs and source-before failures:', fixture)
