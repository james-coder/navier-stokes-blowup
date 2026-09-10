import json
from ns_blowup.observatory import build_observatory


def test_observatory_measured_data_and_precision_experiment(tmp_path):
    path=build_observatory(tmp_path/'observatory.html')
    data=json.loads(path.with_suffix('.json').read_text())
    assert '__DATA__' not in path.read_text()
    assert len(data['tau'])==121
    fp64=[r for r in data['convergence'] if r['precision']=='float64']
    assert fp64[0]['absolute_residual']/fp64[3]['absolute_residual']>40
    fp32=[r for r in data['convergence'] if r['precision']=='float32']
    assert fp32[-1]['absolute_residual']>fp64[-1]['absolute_residual']*100
    assert 'NOT IMPLEMENTED' in data['status']
