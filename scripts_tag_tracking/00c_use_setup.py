"""
Use the setup calibration (scripts_tag_tracking/calibrate_setup.py) in a
session: copies bodies.json and gripper_tcp.json from the config's
setup_calibration directory into <session>/tag_tracking/, with
setup_calibration.json recording the source, so the session keeps the exact
calibration it was processed with. Replaces calibrate_bodies and gripper_tcp.

python scripts_tag_tracking/00c_use_setup.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import shutil
import click
from umi.tag_tracking.config import load_tag_tracking_config


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml with setup_calibration')
def main(input, config):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    assert 'setup_calibration' in config, "tag tracking config needs setup_calibration"
    setup = pathlib.Path(config['setup_calibration'])
    output = session.joinpath('tag_tracking')
    output.mkdir(parents=True, exist_ok=True)
    for name in ['bodies.json', 'gripper_tcp.json']:
        assert setup.joinpath(name).is_file(), f"{setup.joinpath(name)} missing, run calibrate_setup.py"
        shutil.copy2(setup.joinpath(name), output.joinpath(name))
        print(f"Copied {setup.joinpath(name)} -> {output.joinpath(name)}")
    json.dump({'setup_calibration': str(setup)}, output.joinpath('setup_calibration.json').open('w'), indent=2)

# %%
if __name__ == "__main__":
    main()
