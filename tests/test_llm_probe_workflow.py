from pathlib import Path
import re

import yaml


def _workflow():
    return yaml.safe_load(Path('.github/workflows/test.yml').read_text())


def test_llm_probe_is_opt_in_and_excludes_other_test_jobs():
    workflow = _workflow()
    events = workflow.get('on', workflow.get(True))
    probe_input = events['workflow_dispatch']['inputs']['llm_probe']
    assert probe_input['type'] == 'boolean'
    assert probe_input['default'] is False
    jobs = workflow['jobs']
    assert jobs['probe']['if'] == "${{ inputs.llm_probe && inputs.body_replay_run_id == '' }}"
    assert jobs['probe']['timeout-minutes'] == 5
    assert jobs['replay']['if'] == "${{ !inputs.llm_probe && inputs.replay_run_id != '' && inputs.body_replay_run_id == '' }}"
    assert jobs['verify']['if'] == "${{ !inputs.llm_probe && inputs.replay_run_id == '' && inputs.body_replay_run_id == '' }}"
    assert not jobs['probe'].get('needs')


def test_llm_probe_exposes_only_provider_secrets_to_request_step():
    workflow = _workflow()
    probe = workflow['jobs']['probe']
    assert not workflow.get('env')
    assert not probe.get('env')
    assert not probe.get('secrets')
    secrets = set(re.findall(r'secrets\.([A-Za-z_][A-Za-z0-9_]*)', yaml.safe_dump(probe)))
    assert secrets == {'OPENAI_API_KEY', 'OPENAI_API_BASE'}
    steps = probe['steps']
    request = next(step for step in steps if 'scripts/probe_llm.py' in step.get('run', ''))
    assert request['env'] == {
        'OPENAI_API_KEY': '${{ secrets.OPENAI_API_KEY }}',
        'OPENAI_API_BASE': '${{ secrets.OPENAI_API_BASE }}',
    }
    for step in steps:
        if step is not request:
            assert 'secrets.' not in yaml.safe_dump(step)


def test_llm_probe_applies_online_override_and_runs_only_minimal_probe():
    steps = _workflow()['jobs']['probe']['steps']
    assert steps[0]['uses'] == 'actions/checkout@v6'
    assert steps[1]['uses'] == 'astral-sh/setup-uv@v7'
    commands = [step for step in steps if 'run' in step]
    assert [step['run'] for step in commands] == [
        'uv run --frozen scripts/apply_config_override.py',
        'uv run --frozen scripts/probe_llm.py --output verification/llm-probe.json',
    ]
    assert commands[0]['env'] == {'CUSTOM_CONFIG': '${{ vars.CUSTOM_CONFIG }}'}


def test_llm_probe_uploads_only_sanitized_report_even_on_failure():
    steps = _workflow()['jobs']['probe']['steps']
    uploads = [step for step in steps if step.get('uses', '').startswith('actions/upload-artifact@')]
    assert len(uploads) == 1
    upload = uploads[0]
    assert upload['if'] == 'always()'
    assert upload['with']['name'] == 'llm-probe'
    assert upload['with']['path'] == 'verification/llm-probe.json'
    assert upload['with']['retention-days'] == 14
