from absl import app
from absl import flags
import os
import subprocess

_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'unlearning-metric', 'Name of the experiment.', short_name='n'
)
_DATA_DIR = flags.DEFINE_string(
    'data_dir',
    'data/',
    'Directory containing the data.',
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    'checkpoints/',
    'Directory containing the checkpoints.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    'outputs/',
    'Directory to write the outputs.',
)


def main(argv) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Ensure directories exist
    os.makedirs(_DATA_DIR.value, exist_ok=True)
    os.makedirs(_CHECKPOINT_DIR.value, exist_ok=True)
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

    # Define the executable arguments
    executable_args = [
        f"--data_dir={_DATA_DIR.value}",
        f"--checkpoint_dir={_CHECKPOINT_DIR.value}",
        f"--output_dir={_OUTPUT_DIR.value}",
    ]

    # Run the main.py script with the provided arguments
    subprocess.run(["python3", "main.py"] + executable_args, check=True)


if __name__ == '__main__':
    app.run(main)