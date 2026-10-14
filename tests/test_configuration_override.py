from pathlib import Path

from omegaconf import OmegaConf
import pytest

from scripts.apply_config_override import apply_override


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
