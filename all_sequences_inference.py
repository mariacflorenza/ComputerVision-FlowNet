import os
import subprocess
import argparse
import run_inference
# $ python3 all_sequences_inference.py --pretrained flownets_EPE1.951.pth --mode direct/sequential
# Pretrained model path from the arguments
parser = argparse.ArgumentParser(description="Run inference for all sequences using FlowNet")
parser.add_argument('--pretrained', type=str, required=True, help='Path to the pre-trained model')
parser.add_argument('--mode', type=str, default='sequential', choices=['sequential', 'direct', 'inference'],
                        help='Choose mode: complete_inferece_saving_seq, inference_direct, or inference')
args = parser.parse_args()

sequence_list = ['bear', 'book', 'bag', 'camel', 'rhino', 'swan']

run_inference_script = 'run_inference_test.py'

pretrained_model_path = args.pretrained

image_folder = 'sequences-train'

for sequence in sequence_list:
    # image_folder = os.path.join(image_folder_base, sequence)
    command = [
        'python3', run_inference_script,
        '--sequences_path', image_folder,
        '--model_path', pretrained_model_path,
        '--mode', args.mode,
        '--sequence', sequence,
    ]
    print(f"Running inference for sequence: {sequence}")
    subprocess.run(command)
    print("-"*50)
    print("\n")