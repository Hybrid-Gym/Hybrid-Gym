"""Build the masked docstring-locate dataset from SWE-Gym raw instances.

This variant generates data where:
1. The function/class name is NOT mentioned in the description (for true masked evaluation)
2. Detailed information is extracted: parameters, return values, function calls
3. Multiple levels of description detail for tunable difficulty

Features:
- Incremental saving: Saves progress after each function is processed
- Resumable: Skips already-processed functions based on unique identifiers
- Structured JSON output from LLM for reliable parsing
"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import subprocess
import warnings
from typing import Any, List, Optional, Sequence, Tuple
import threading

# Suppress SyntaxWarnings from AST parsing old Python code with deprecated escape sequences
warnings.filterwarnings("ignore", category=SyntaxWarning)

from datasets import load_dataset
from tqdm import tqdm

HOME_DIR = os.environ.get("HOME", "")
DOWNLOAD_DIR = os.path.join(HOME_DIR, "tmp/")
CODE_DIR = os.path.join(HOME_DIR, "OpenHands2/")

# Thread-safe lock for file writes
_write_lock = threading.Lock()


def get_function_unique_id(repo: str, file_path: str, module_name: str, line_start: int) -> str:
    """Generate a unique identifier for a function based on repo, file, name, and line."""
    key = f"{repo}::{file_path}::{module_name}::{line_start}"
    return hashlib.md5(key.encode()).hexdigest()


def load_processed_ids(output_path: str) -> set:
    """Load already-processed function IDs from existing output file."""
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        func_id = get_function_unique_id(
                            entry.get("repo", ""),
                            entry.get("file_path", ""),
                            entry.get("module_name", ""),
                            entry.get("module_line_start", 0),
                        )
                        processed_ids.add(func_id)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[warn] Error loading existing data: {e}. Starting fresh.")
    return processed_ids


def append_to_jsonl(output_path: str, entry: dict) -> None:
    """Thread-safe append a single entry to JSONL file."""
    with _write_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def clone_and_checkout(repo_name: str, commit: str, repo_dir: str) -> None:
    """Clone repo if needed and checkout the given commit."""
    if not os.path.exists(repo_dir):
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo_name}.git", repo_dir],
            check=True,
        )
    subprocess.run(["git", "-C", repo_dir, "checkout", commit], check=True)


def remove_repo_dir(repo_dir: str) -> None:
    """Remove cloned repository directory to save disk space."""
    if os.path.exists(repo_dir):
        subprocess.run(["rm", "-rf", repo_dir], check=True)
        print(f"[info] Removed repo dir: {repo_dir}")


def sample_dataset(dataset, sample_size: Optional[int], seed: int):
    """Sample dataset to a specified size."""
    if not sample_size or sample_size <= 0:
        return dataset
    sample_size = min(sample_size, len(dataset))
    if isinstance(dataset, list):
        rng = random.Random(seed)
        rng.shuffle(dataset)
        return dataset[:sample_size]
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return dataset.select(indices[:sample_size])


def extract_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    """Extract function signature details using AST."""
    parameters = []

    # Handle regular args
    args = node.args
    defaults_offset = len(args.args) - len(args.defaults)

    for i, arg in enumerate(args.args):
        param = {"name": arg.arg}

        # Get type annotation if present
        if arg.annotation:
            try:
                param["type"] = ast.unparse(arg.annotation)
            except:
                param["type"] = "Any"

        # Get default value if present
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            try:
                default_val = ast.unparse(args.defaults[default_idx])
                param["default"] = default_val
            except:
                param["default"] = "..."

        # Skip 'self' and 'cls' for methods
        if arg.arg not in ('self', 'cls'):
            parameters.append(param)

    # Handle *args
    if args.vararg:
        param = {"name": f"*{args.vararg.arg}"}
        if args.vararg.annotation:
            try:
                param["type"] = ast.unparse(args.vararg.annotation)
            except:
                pass
        parameters.append(param)

    # Handle keyword-only args
    kw_defaults_map = {i: d for i, d in enumerate(args.kw_defaults) if d is not None}
    for i, arg in enumerate(args.kwonlyargs):
        param = {"name": arg.arg}
        if arg.annotation:
            try:
                param["type"] = ast.unparse(arg.annotation)
            except:
                pass
        if i in kw_defaults_map:
            try:
                param["default"] = ast.unparse(kw_defaults_map[i])
            except:
                param["default"] = "..."
        parameters.append(param)

    # Handle **kwargs
    if args.kwarg:
        param = {"name": f"**{args.kwarg.arg}"}
        if args.kwarg.annotation:
            try:
                param["type"] = ast.unparse(args.kwarg.annotation)
            except:
                pass
        parameters.append(param)

    return {"parameters": parameters}


def extract_return_info(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    """Extract return type annotation if present."""
    if node.returns:
        try:
            return {"returns": {"type": ast.unparse(node.returns)}}
        except:
            pass
    return {"returns": None}


def extract_function_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    """Extract function calls made within the function body."""
    calls = []

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, call_node):
            call_info = {}
            # Get the function name being called
            if isinstance(call_node.func, ast.Name):
                call_info["function"] = call_node.func.id
            elif isinstance(call_node.func, ast.Attribute):
                # Handle method calls like obj.method()
                try:
                    call_info["function"] = ast.unparse(call_node.func)
                except:
                    call_info["function"] = call_node.func.attr
            else:
                try:
                    call_info["function"] = ast.unparse(call_node.func)
                except:
                    call_info["function"] = "<unknown>"

            if call_info.get("function") and call_info["function"] not in [c.get("function") for c in calls]:
                calls.append(call_info)

            self.generic_visit(call_node)

    visitor = CallVisitor()
    for child in node.body:
        visitor.visit(child)

    # Limit to first 10 unique calls to avoid excessive data
    return {"call_details": calls[:10]}


def extract_class_info(node: ast.ClassDef) -> dict:
    """Extract class information including bases and methods."""
    info = {
        "bases": [],
        "methods": []
    }

    # Get base classes
    for base in node.bases:
        try:
            info["bases"].append(ast.unparse(base))
        except:
            pass

    # Get method names
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info["methods"].append(item.name)

    return info


def get_detailed_node_info(filepath: str, line_nums: Sequence[int]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[str], Optional[dict]]:
    """Find the function/class and extract detailed information."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(f"error parsing {filepath}: {exc}")
        return None, None, None, None, None, None, None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        node_type = "class" if isinstance(node, ast.ClassDef) else "function"
        docstring = ast.get_docstring(node)
        if not docstring:
            continue

        module_start_line = node.lineno - 1
        module_end_line = getattr(node, "end_lineno", module_start_line) - 1
        doc_node = node.body[0]
        if isinstance(doc_node, ast.Expr) and isinstance(doc_node.value, ast.Constant):
            doc_start = doc_node.lineno - 1
            doc_end = getattr(doc_node, "end_lineno", doc_start) - 1
        else:
            continue

        for line in line_nums:
            if module_start_line <= line <= module_end_line:
                # Extract detailed information based on node type
                detailed_info = {}

                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    detailed_info.update(extract_function_signature(node))
                    detailed_info.update(extract_return_info(node))
                    detailed_info.update(extract_function_calls(node))
                else:  # ClassDef
                    detailed_info.update(extract_class_info(node))

                # Get source code (without docstring for context)
                source_lines = source.split('\n')
                func_source = '\n'.join(source_lines[module_start_line:module_end_line + 1])

                return (
                    node.name,
                    node_type,
                    docstring,
                    (module_start_line, module_end_line),
                    (doc_start, doc_end),
                    func_source,
                    detailed_info,
                )
    return None, None, None, None, None, None, None


def generate_masked_description(
    docstring: str,
    module_name: str,
    module_type: str,
    detailed_info: dict,
    source_code: str,
    model: str,
    temperature: float,
) -> dict:
    """Use an LLM to produce a description WITHOUT mentioning the function/class name.

    Returns a structured dict with:
    - brief_description: High-level description (no name mentioned)
    - parameters: List of parameter info with descriptions
    - returns: Return value info with description
    - call_details: Function calls with purposes
    """
    if not docstring:
        return {}

    # Build the prompt for structured JSON output
    prompt = f"""You are preparing a masked docstring-location dataset. Your task is to analyze a Python {module_type} and generate structured information about it.

CRITICAL RULES:
1. NEVER mention the {module_type} name "{module_name}" in any description
2. Refer to it as "this {module_type}" or "the {module_type}" instead
3. Do not quote the docstring verbatim; summarize in your own words

Here is the {module_type} information:

Original docstring:
{docstring}

Source code:
{source_code}

Known signature details:
{json.dumps(detailed_info, indent=2)}

Generate a JSON response with the following structure:
{{
    "brief_description": "A 1-3 sentence high-level description of what this {module_type} does. Do NOT include the name '{module_name}'.",
    "parameters": [
        {{"name": "param_name", "type": "param_type or null", "default": "default_value or null", "description": "What this parameter is for"}}
    ],
    "returns": {{
        "type": "return_type or null",
        "description": "What the {module_type} returns"
    }},
    "call_details": [
        {{"function": "function_name", "purpose": "Why this function is called"}}
    ]
}}

If parameters, returns, or call_details are not applicable or unknown, use empty lists/null.
Only include the JSON in your response, no other text."""

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content.strip()

        # Parse the JSON response
        try:
            result = json.loads(response_text)

            # Validate that the name is not in the brief_description
            brief = result.get("brief_description", "")
            if module_name.lower() in brief.lower():
                # Try to remove the name if it slipped through
                brief = re.sub(
                    rf'\b{re.escape(module_name)}\b',
                    f"this {module_type}",
                    brief,
                    flags=re.IGNORECASE
                )
                result["brief_description"] = brief

            return result
        except json.JSONDecodeError as e:
            print(f"[warn] Failed to parse LLM JSON response: {e}")
            # Fall back to basic structure
            return {
                "brief_description": docstring.strip().split("\n")[0],
                "parameters": detailed_info.get("parameters", []),
                "returns": detailed_info.get("returns"),
                "call_details": detailed_info.get("call_details", []),
            }

    except Exception as exc:
        print(f"[warn] LLM call failed ({exc}), falling back to basic extraction")
        # Fall back to the first sentence of the docstring
        first_sentence = docstring.strip().split("\n")[0]
        # Remove any mention of the function name from the first sentence
        first_sentence = re.sub(
            rf'\b{re.escape(module_name)}\b',
            f"this {module_type}",
            first_sentence,
            flags=re.IGNORECASE
        )
        return {
            "brief_description": first_sentence,
            "parameters": detailed_info.get("parameters", []),
            "returns": detailed_info.get("returns"),
            "call_details": detailed_info.get("call_details", []),
        }


def get_all_func_w_docstrings(repo_dir: str) -> list[tuple[str, int]]:
    """Return (file_path, line_no) pairs for every function/class that has a docstring."""
    results: list[tuple[str, int]] = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if ast.get_docstring(node):
                            results.append((file_path, node.lineno))
            except (SyntaxError, UnicodeDecodeError, FileNotFoundError, OSError):
                continue
    return results


def build_masked_dataset(
    dataset,
    output_path: str,
    model: str,
    temperature: float,
    sample_size: Optional[int],
    seed: int = 42,
) -> list[dict]:
    """Build the masked dataset with detailed function information.

    Args:
        dataset: Source dataset (HuggingFace or list)
        output_path: Path to output JSONL file
        model: OpenAI model name
        temperature: Sampling temperature
        sample_size: Number of instances to process (0 = all)
        seed: Random seed

    Returns:
        List of processed instances
    """
    # Load already processed IDs for resume support
    processed_ids = load_processed_ids(output_path)
    print(f"[info] Found {len(processed_ids)} already processed functions. Resuming...")

    # Select unique repos
    selected_instances = []
    repos = []
    for instance in tqdm(dataset, total=len(dataset), desc="Selecting repos"):
        if instance["repo"] not in repos:
            selected_instances.append(instance)
            repos.append(instance["repo"])

    selected_instances = sample_dataset(selected_instances, sample_size, seed=seed)

    stats = {
        "total_processed": 0,
        "successful": 0,
        "skipped_already_done": 0,
        "failed": 0,
    }

    for instance in tqdm(selected_instances, total=len(selected_instances), desc="Processing repos"):
        repo_name = instance["repo"]
        repo_dir = os.path.join(DOWNLOAD_DIR, repo_name.split("/")[-1])

        try:
            clone_and_checkout(repo_name, instance["base_commit"], repo_dir)
        except subprocess.CalledProcessError as exc:
            print(f"[warn] Skip repo {repo_name} due to checkout failure: {exc}")
            continue

        # Get all functions with docstrings
        loc_pairs = get_all_func_w_docstrings(repo_dir)

        for module_idx, (full_path, line_num) in tqdm(
            enumerate(loc_pairs),
            total=len(loc_pairs),
            desc=f"{repo_name}",
            leave=False,
        ):
            if not os.path.exists(full_path):
                continue

            # Get detailed node info
            result = get_detailed_node_info(full_path, [line_num])
            (
                module_name,
                module_type,
                docstring,
                module_line_range,
                docstring_line_range,
                source_code,
                detailed_info,
            ) = result

            if module_name is None:
                stats["failed"] += 1
                continue

            file_path = full_path.split(repo_dir + "/")[1]

            # Check if already processed
            func_id = get_function_unique_id(
                instance["repo"],
                file_path,
                module_name,
                module_line_range[0],
            )

            if func_id in processed_ids:
                stats["skipped_already_done"] += 1
                continue

            # Generate masked description with detailed info
            masked_info = generate_masked_description(
                docstring=docstring,
                module_name=module_name,
                module_type=module_type,
                detailed_info=detailed_info,
                source_code=source_code,
                model=model,
                temperature=temperature,
            )

            # Build the new instance
            new_instance = {
                # Core instance info
                "instance_id": f"{instance['instance_id']}_{module_idx}",
                "repo": instance["repo"],
                "base_commit": instance["base_commit"],
                "image_instance_id": instance.get("instance_id", instance.get("image_instance_id")),

                # File and location info
                "file_path": file_path,
                "module_name": module_name,  # Stored but NOT used in prompts
                "module_type": module_type,
                "module_line_start": module_line_range[0],
                "module_line_end": module_line_range[1],

                # Docstring info (for removal during runtime)
                "docstring": docstring,
                "docstring_line_start": docstring_line_range[0],
                "docstring_line_end": docstring_line_range[1],

                # Masked description (does NOT contain function name)
                "brief_description": masked_info.get("brief_description", ""),

                # Detailed information for tunable difficulty
                "parameters": masked_info.get("parameters", []),
                "returns": masked_info.get("returns"),
                "call_details": masked_info.get("call_details", []),

                # Raw AST-extracted info (backup)
                "ast_info": detailed_info,
            }

            # Save incrementally
            append_to_jsonl(output_path, new_instance)
            processed_ids.add(func_id)
            stats["successful"] += 1
            stats["total_processed"] += 1

        # Clean up repo
        remove_repo_dir(repo_dir)

    print(f"\n[info] Processing complete:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Skipped (already done): {stats['skipped_already_done']}")
    print(f"  Failed: {stats['failed']}")

    # Load and return all data
    all_data = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))

    return all_data


def main():
    parser = argparse.ArgumentParser(
        description="Build masked docstring-locate dataset with detailed function info"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SWE-Gym/SWE-Gym-Raw",
        help="Base dataset to build from (default: SWE-Gym/SWE-Gym-Raw).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of repos to process (0 = all).",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="Model used to synthesize function descriptions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for description generation.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL file path. Defaults to resource/swe_doc_gen_masked_{dataset}_{split}.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset_save_name = args.dataset_name.split("/")[-1]

    # Set output path
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(
            CODE_DIR,
            f"evaluation/benchmarks/hybrid_gym_func_localize/resource/swe_doc_gen_masked_{dataset_save_name}_{args.split}.jsonl",
        )

    print(f"[info] Output path: {output_path}")
    print(f"[info] Dataset: {args.dataset_name}, split: {args.split}")
    print(f"[info] Sample size: {args.sample_size if args.sample_size > 0 else 'all'}")

    # Build the dataset
    build_masked_dataset(
        dataset=dataset,
        output_path=output_path,
        model=args.openai_model,
        temperature=args.temperature,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
