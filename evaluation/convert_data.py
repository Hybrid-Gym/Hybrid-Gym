#!/usr/bin/env python3
"""
Convert evaluation output data to training format.
Processes gzipped JSONL files containing evaluation results and converts them to a format suitable for training.
"""

import argparse
import copy
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openhands.llm.fn_call_converter import (
    FunctionCallConversionError,
    convert_fncall_messages_to_non_fncall_messages,
    convert_from_multiple_tool_calls_to_single_tool_call_messages,
)

tqdm.pandas()

STORAGE_DIR = os.environ.get('STORAGE_DIR', '/projects/ogma3/yiqingxi')


def parse_git_diff(diff_text):
    """Parse git diff to extract added lines."""
    lines_added = []

    for line in diff_text.splitlines():
        if line.startswith('+++') or line.startswith('---'):
            continue  # skip file metadata
        if line.startswith('+'):
            lines_added.append(line[1:])  # strip the '+' sign

    return lines_added


def judge_valid_docstring_patch(diff_text):
    """Check if git patch contains valid docstring additions."""
    lines_added = parse_git_diff(diff_text)

    if len(lines_added) == 0:
        return False

    block = "\n".join(lines_added).strip()
    if block.endswith('"""') or block.endswith("'''") or block.endswith("```"):
        return True
    return False


def _contains_multiple_tool_calls(messages: list[dict]) -> bool:
    """Check if any message contains multiple tool calls."""
    return any(
        message.get('tool_calls') and len(message['tool_calls']) > 1
        for message in messages
    )


def _convert_messages(messages: list[dict], tools: list[dict], failed_counter: dict) -> list[dict]:
    """Convert function calling messages to non-function calling format."""
    message_copy = copy.deepcopy(messages)
    for message in message_copy:
        if message['content'] is None:
            message['content'] = ''

    try:
        message_converted = convert_fncall_messages_to_non_fncall_messages(
            message_copy, tools, add_in_context_learning_example=False
        )
        message_clean = []
        for message in message_converted:
            if type(message["content"]) == list and type(message['content'][0]) == dict:
                message['content'] = message['content'][0]['text']
            assert type(message['content']) == str
            message_clean.append(message)
        return message_clean
    except FunctionCallConversionError:
        failed_counter['count'] += 1
        return None


def load_data_from_files(file_paths: list[str]) -> pd.DataFrame:
    """Load and process data from gzipped JSONL files."""
    data = []
    length_list = []
    existing_instance_ids = {}
    repeat_id_count = 0

    for FILE_PATH in file_paths:
        with gzip.open(FILE_PATH, 'rb') as f:
            for i, line in tqdm(
                enumerate(f), desc=f'Processing {os.path.basename(FILE_PATH)}'
            ):
                raw_data = json.loads(line)

                # Handle duplicate instance IDs
                if raw_data['instance_id'] in existing_instance_ids:
                    repeat_id_count += 1
                    if "func_gen_outputs" not in FILE_PATH or existing_instance_ids[raw_data['instance_id']] >= 2:
                        continue

                # Skip if no raw completions
                if "raw_completions" not in raw_data or raw_data['raw_completions'] is None:
                    continue
                if "messages" not in raw_data['raw_completions']:
                    continue

                # Track instance IDs
                if raw_data['instance_id'] not in existing_instance_ids:
                    existing_instance_ids[raw_data['instance_id']] = 0
                existing_instance_ids[raw_data['instance_id']] += 1

                msg_len = len(raw_data['raw_completions']['messages'])
                length_list.append(msg_len)

                data.append({
                    'resolved': True,
                    'messages': raw_data['raw_completions']['messages']
                    if raw_data['raw_completions'] is not None
                    and len(raw_data['raw_completions']['messages']) > 0 else None,
                    'git_patch': raw_data['test_result'].get('git_patch', ''),
                    'tools': raw_data['raw_completions']['tools']
                    if raw_data['raw_completions'] is not None
                    and 'tools' in raw_data['raw_completions']
                    else None,
                    })

    df = pd.DataFrame(data)
    print(f'#total amount of data={len(df)}')
    df = df[~df['messages'].isna()]
    print(f'#total amount of data after removing nan={len(df)}')
    print(f'#total amount of repeat ids={repeat_id_count}')
    if length_list:
        print(f'#average agent steps={np.mean(length_list)}')

    return df


def process_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Convert messages to non-function calling format."""
    # Check for multiple tool calls
    df['contains_multiple_tool_calls'] = df['messages'].apply(_contains_multiple_tool_calls)

    # Convert messages
    failed_counter = {'count': 0}

    df['converted_messages'] = df.apply(
        lambda row: convert_from_multiple_tool_calls_to_single_tool_call_messages(
            row['messages'], ignore_final_tool_result=True
        ),
        axis=1,
    )

    df['nonfncall_messages'] = df.apply(
        lambda row: _convert_messages(row['converted_messages'], row['tools'], failed_counter),
        axis=1
    )

    print('total nan', df['nonfncall_messages'].isna().sum())
    df = df[~df['nonfncall_messages'].isna()]
    print(f'Total failed: {failed_counter["count"]}')

    return df


def save_and_upload(df: pd.DataFrame, file_path: str, push_to_hub: bool = False) -> str:
    """Save processed data and optionally push to HuggingFace hub."""
    folder_name = "hybrid_gym_outputs"

    output_dir = os.path.join(
        f'{STORAGE_DIR}/openhands/evaluation/evaluation_outputs/{folder_name}'
    )
    os.makedirs(output_dir, exist_ok=True)

    # Extract dataset and model name from path
    path_parts = file_path.split('/')
    dataset_name = "unknown"
    model_name = "unknown"

    for part in path_parts:
        if "CodeActAgent" in path_parts:
            idx = path_parts.index("CodeActAgent")
            if idx > 0:
                dataset_name = path_parts[idx - 1].replace('__', '_').replace('-', '_')
            if idx + 1 < len(path_parts):
                model_part = path_parts[idx + 1]
                model_name = model_part.split('_maxiter')[0]

    num_resolved = df[df['resolved']]['resolved'].count()
    output_file = os.path.join(
        output_dir,
        f'{dataset_name}_{model_name}_{num_resolved}i.jsonl',
    )

    # Save to JSONL
    df[df['resolved']][['nonfncall_messages']].rename(
        columns={'nonfncall_messages': 'messages'}
    ).to_json(
        output_file,
        lines=True,
        orient='records',
    )

    print(f'Saved to: {output_file}')
    print(f'Total resolved instances: {num_resolved}')

    # Push to HuggingFace hub if requested
    if push_to_hub:
        save_name = os.path.basename(output_file).split('.')[0]
        data = [json.loads(line) for line in open(output_file)]
        dataset = Dataset.from_list(data)
        dataset.push_to_hub(f"hybrid-gym/{save_name}")
        print(f"Dataset pushed to hub: hybrid-gym/{save_name}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Convert evaluation output data to training format'
    )
    parser.add_argument(
        '--file_paths',
        type=str,
        required=True,
        help='Paths to gzipped JSONL files for successful trajectories'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push processed dataset to HuggingFace hub'
    )
    parser.add_argument(
        '--storage-dir',
        default=None,
        help='Override STORAGE_DIR environment variable'
    )

    args = parser.parse_args()

    # Override STORAGE_DIR if provided
    if args.storage_dir:
        global STORAGE_DIR
        STORAGE_DIR = args.storage_dir

    file_paths = args.file_paths.split(',')

    print(f"Processing {len(file_paths)} file(s)...")
    print(f"Storage directory: {STORAGE_DIR}")

    # Load data
    df = load_data_from_files(file_paths)

    # Process messages
    df = process_messages(df)

    # Print statistics
    print(f'\n#total amount of valid data: {df["resolved"].sum()}')

    # Save and optionally upload
    output_file = save_and_upload(df, file_paths[0], args.push_to_hub)

    print(f'\nProcessing complete!')
    print(f'Output file: {output_file}')


if __name__ == '__main__':
    main()
