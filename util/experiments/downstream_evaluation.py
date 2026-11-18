"""Launch BrainBERT checkpoint evaluations with scoped test-window defaults.

Run with a brain_decoding single-task YAML, its checkout, an output directory
and a shared manifest directory. The task name is the YAML stem. The generated
configuration records the explicit evaluation policy. Other benchmark entry
points retain their original full-test behaviour. Pass --full-test to opt out.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

POLICY_PATH = Path(__file__).resolve().parents[2] / (
    'conf/evaluation/downstream.yaml'
)


def configure_evaluation(
    config: dict, task: str, manifests: Path, full_test: bool = False,
) -> dict:
    """Apply the BrainBERT policy to a task configuration in place.

    Parameters
    ----------
    config : dict
        Existing brain_decoding single-task configuration.
    task : str
        Benchmark task identifier.
    manifests : Path
        Shared model-independent test manifest directory.
    full_test : bool, optional
        Explicitly retain all test windows, default False.

    Returns
    -------
    dict
        Configuration with a scoped opt-in test policy.
    """
    policy = yaml.safe_load(POLICY_PATH.read_text())
    params = config['trainer']['params']
    params.pop('test_subset', None)
    if not full_test and task in policy['limited_tasks']:
        params['test_subset'] = {
            'max_windows': policy['max_windows'],
            'block_size': policy['block_size'],
            'manifest_root': str(manifests.resolve() / task),
        }
        params['prediction_batch_size'] = policy['prediction_batch_size']
    return config


def main() -> None:
    """Compose a separate invocation and use the shared benchmark runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifest-root', type=Path, required=True)
    parser.add_argument('--full-test', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config = configure_evaluation(
        config, args.config.stem, args.manifest_root, args.full_test,
    )
    source = Path(config['trainer']['params']['log_dir']).resolve()
    if (source / 'component_fits').is_dir():
        config['trainer']['params']['fit_source_root'] = str(source)
        config['trainer']['params']['require_completed_fits'] = True
    output = args.output_root.resolve()
    if output == Path(config['log_dir']).resolve():
        raise ValueError('Use a separate output root for this evaluation.')

    def relocate(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'log_dir':
                    value[key] = str(output)
                else:
                    relocate(item)
        elif isinstance(value, list):
            for item in value:
                relocate(item)

    relocate(config)
    output.mkdir(parents=True, exist_ok=True)
    invocation = output / 'evaluation_config.yaml'
    text = yaml.safe_dump(config, sort_keys=False)
    if invocation.exists() and invocation.read_text() != text:
        raise ValueError('Existing evaluation configuration differs.')
    invocation.write_text(text)
    subprocess.run(
        [sys.executable, str(args.checkout / 'training_main.py'),
         'single', str(invocation), '--verbose', '0'],
        cwd=args.checkout, check=True,
    )


if __name__ == '__main__':
    main()
