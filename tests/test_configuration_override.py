from pathlib import Path

from omegaconf import OmegaConf
import pytest

from scripts.apply_config_override import apply_override


@pytest.fixture
def recommendation_configs(tmp_path):
    defaults = {
        'reranker': {
            'relevance': {'policy': 'reading_value', 'top_k': 60, 'profile': 'branch profile',
                          'shortlist': {'embedding_k': 40, 'core_k': 20}},
            'ranking': {'mode': 'theme_balanced'},
            'local': {'model': 'branch-embedding'},
        },
        'executor': {'max_paper_num': 8, 'history_enabled': True, 'progress_dir': '.paper-progress'},
        'llm': {'summary_instructions': 'branch instructions', 'generation_kwargs': {'model': 'branch-model'},
                'api': {'key': '${oc.env:OPENAI_API_KEY}', 'base_url': '${oc.env:OPENAI_API_BASE}'}},
        'zotero': {'api_key': '${oc.env:ZOTERO_KEY}', 'include_path': ['branch/**']},
        'email': {'receiver': '${oc.env:RECEIVER}', 'sender_password': '${oc.env:SENDER_PASSWORD}'},
    }
    override = {
        'reranker': {
            'relevance': {'policy': 'score', 'top_k': 30, 'profile': 'online profile',
                          'shortlist': {'obsolete_setting': True}, 'old_only': True},
            'ranking': {'mode': 'legacy'},
            'local': {'model': 'online-embedding'},
        },
        'executor': {'max_paper_num': 10, 'history_enabled': False, 'progress_dir': 'online-progress',
                     'legacy_history_scopes': [{'include': ['online/**']}]},
        'llm': {'summary_instructions': 'online instructions', 'generation_kwargs': {'model': 'online-model'},
                'api': {'key': '${oc.env:ONLINE_KEY}', 'base_url': '${oc.env:ONLINE_BASE}'}},
        'zotero': {'api_key': '${oc.env:ONLINE_ZOTERO_KEY}', 'include_path': ['online/**']},
        'email': {'receiver': '${oc.env:ONLINE_RECEIVER}', 'sender_password': '${oc.env:ONLINE_PASSWORD}'},
        'source': {'arxiv': {'category': ['physics.ins-det'], 'include_cross_list': False}},
    }
    path = tmp_path / 'custom.yaml'
    OmegaConf.save(OmegaConf.create(defaults), path, resolve=False)
    return path, defaults, override


def test_repository_recommendations_exactly_replace_selected_settings(recommendation_configs):
    path, defaults, override = recommendation_configs
    apply_override(path, OmegaConf.to_yaml(OmegaConf.create(override)), repository_recommendations=True)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert raw['reranker']['relevance'] == defaults['reranker']['relevance']
    assert raw['executor']['max_paper_num'] == defaults['executor']['max_paper_num']
    assert raw['llm']['summary_instructions'] == defaults['llm']['summary_instructions']
    # Every unselected setting, including environment references and delivery scope, stays online.
    raw['reranker'].pop('relevance')
    override['reranker'].pop('relevance')
    raw['executor'].pop('max_paper_num')
    override['executor'].pop('max_paper_num')
    raw['llm'].pop('summary_instructions')
    override['llm'].pop('summary_instructions')
    assert raw == override


def test_default_merge_keeps_explicit_online_recommendations(recommendation_configs):
    path, defaults, override = recommendation_configs
    apply_override(path, OmegaConf.to_yaml(OmegaConf.create(override)))
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert raw['reranker']['relevance']['policy'] == 'score'
    assert raw['reranker']['relevance']['top_k'] == 30
    assert raw['reranker']['relevance']['old_only'] is True
    assert raw['reranker']['relevance']['shortlist'] == {
        **defaults['reranker']['relevance']['shortlist'], 'obsolete_setting': True}
    assert raw['executor']['max_paper_num'] == 10
    assert raw['llm']['summary_instructions'] == 'online instructions'


@pytest.mark.parametrize('dotted', ['reranker.relevance', 'executor.max_paper_num', 'llm.summary_instructions'])
@pytest.mark.parametrize('missing_value', ['absent', None, '???'])
def test_repository_recommendations_missing_default_never_writes(recommendation_configs, dotted, missing_value):
    path, defaults, override = recommendation_configs
    parent, key = dotted.split('.')
    if missing_value == 'absent':
        defaults[parent].pop(key)
    else:
        defaults[parent][key] = missing_value
    OmegaConf.save(OmegaConf.create(defaults), path, resolve=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='Repository configuration must define'):
        apply_override(path, OmegaConf.to_yaml(OmegaConf.create(override)), repository_recommendations=True)
    assert path.read_bytes() == original


def test_repository_recommendations_still_reject_literal_credentials(recommendation_configs):
    path, _, override = recommendation_configs
    original = path.read_bytes()
    override['llm']['api']['key'] = 'SECRET_SENTINEL_MUST_NOT_ECHO'
    with pytest.raises(ValueError) as caught:
        apply_override(path, OmegaConf.to_yaml(OmegaConf.create(override)), repository_recommendations=True)
    assert 'SECRET_SENTINEL_MUST_NOT_ECHO' not in str(caught.value)
    assert path.read_bytes() == original


def test_repository_recommendations_cli_uses_checked_out_settings(recommendation_configs):
    import os
    import subprocess
    import sys

    path, defaults, override = recommendation_configs
    config_dir = path.parent / 'config'
    config_dir.mkdir()
    path.rename(config_dir / 'custom.yaml')
    result = subprocess.run(
        [sys.executable, str(Path('scripts/apply_config_override.py').resolve()), '--repository-recommendations'],
        cwd=path.parent,
        env={**os.environ, 'CUSTOM_CONFIG': OmegaConf.to_yaml(OmegaConf.create(override))},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'online profile' not in result.stdout + result.stderr
    raw = OmegaConf.to_container(OmegaConf.load(config_dir / 'custom.yaml'), resolve=False)
    assert raw['reranker']['relevance'] == defaults['reranker']['relevance']
    assert raw['llm']['api'] == override['llm']['api']


def test_workflow_repository_recommendations_only_affect_explicit_dry_run():
    import yaml

    workflow = yaml.safe_load(Path('.github/workflows/main.yml').read_text())
    # PyYAML's YAML 1.1 reader represents the Actions "on" key as True.
    events = workflow.get('on', workflow.get(True))
    inputs = events['workflow_dispatch']['inputs']
    assert inputs['repository_recommendations']['type'] == 'boolean'
    assert inputs['repository_recommendations']['default'] is False
    assert inputs['dry_run']['default'] is False
    steps = workflow['jobs']['calculate-and-send']['steps']
    override_steps = [step for step in steps if 'scripts/apply_config_override.py' in step.get('run', '')]
    assert len(override_steps) == 1
    assert override_steps[0]['env']['CUSTOM_CONFIG'] == '${{ vars.CUSTOM_CONFIG }}'
    assert override_steps[0]['run'] == (
        "uv run --frozen scripts/apply_config_override.py "
        "${{ inputs.dry_run && inputs.repository_recommendations && '--repository-recommendations' || '' }}")
    process = next(step for step in steps if step.get('id') == 'process')
    assert "executor.dry_run=${{ inputs.dry_run && 'true' || 'false' }}" in process['run']


def test_partial_override_retains_new_defaults_and_explicit_choices(tmp_path):
    path = tmp_path / 'custom.yaml'
    path.write_text('''reranker:
  ranking:
    mode: theme_balanced
executor:
  history_enabled: true
  progress_dir: .paper-progress
llm:
  api:
    key: ${oc.env:OPENAI_API_KEY}
  generation_kwargs:
    model: original
''')
    apply_override(path, 'llm:\n  generation_kwargs:\n    model: chosen\n')
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert raw['reranker']['ranking']['mode'] == 'theme_balanced'
    assert raw['executor']['history_enabled'] is True
    assert raw['llm']['generation_kwargs']['model'] == 'chosen'
    assert raw['llm']['api']['key'] == '${oc.env:OPENAI_API_KEY}'
    apply_override(path, 'executor:\n  history_enabled: false\nreranker:\n  ranking:\n    mode: legacy\n')
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert raw['executor']['history_enabled'] is False
    assert raw['reranker']['ranking']['mode'] == 'legacy'


@pytest.mark.parametrize('value', ['sk-secret-value', '${oc.env:KEY,sk-secret-value}'])
def test_literal_credentials_rejected_without_writing_or_echoing(tmp_path, value):
    path = tmp_path / 'custom.yaml'
    original = 'llm:\n  api:\n    key: ${oc.env:KEY}\n'
    path.write_text(original)
    with pytest.raises(ValueError) as caught:
        apply_override(path, 'llm:\n  api:\n    key: ' + value)
    assert 'sk-secret-value' not in str(caught.value)
    assert path.read_text() == original


def test_list_config_is_rejected_before_write(tmp_path):
    path = tmp_path / 'custom.yaml'
    path.write_text('executor:\n  history_enabled: true\n')
    with pytest.raises(ValueError, match='YAML mapping'):
        apply_override(path, '- arxiv\n- biorxiv')


def test_malformed_override_cli_never_echoes_sensitive_yaml(tmp_path):
    import os
    import subprocess
    import sys
    path = tmp_path / 'config'
    path.mkdir()
    (path / 'custom.yaml').write_text('executor:\n  history_enabled: true\n')
    result = subprocess.run([sys.executable, str(Path('scripts/apply_config_override.py').resolve())], cwd=tmp_path,
                            env={**os.environ, 'CUSTOM_CONFIG': 'llm: [ SECRET_SENTINEL_MUST_NOT_ECHO'},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Configuration override failed' in result.stderr
    assert 'SECRET_SENTINEL_MUST_NOT_ECHO' not in result.stdout + result.stderr
