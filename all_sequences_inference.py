import os
import subprocess
import argparse
import run_inference
# $ python3 all_sequences_inference.py --pretrained flownets_EPE1.951.pth
# Pretrained model path from the arguments
parser = argparse.ArgumentParser(description="Run inference for all sequences using FlowNet")
parser.add_argument('--pretrained', type=str, required=True, help='Path to the pre-trained model')
args = parser.parse_args()

sequence_list = ['bear', 'book', 'bag', 'camel', 'rhino', 'swan']

run_inference_script = 'run_inference.py'

pretrained_model_path = args.pretrained

image_folder = 'sequences-train'

for sequence in sequence_list:
    # image_folder = os.path.join(image_folder_base, sequence)
    command = [
        'python3', run_inference_script,
        '--data', image_folder,
        '--pretrained', pretrained_model_path,
        '--sequence', sequence,
    ]
    print(f"Running inference for sequence: {sequence}")
    subprocess.run(command)