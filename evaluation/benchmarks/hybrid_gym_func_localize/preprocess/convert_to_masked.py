"""Convert existing swe_doc_gen_locate data to masked format.

This script takes existing data (with function names in descriptions) and:
1. Regenerates descriptions WITHOUT mentioning the function/class name
2. Adds detailed parameter, return, and call information
3. Uses the existing repos (cloned on-demand) to extract AST info

Usage:
    python convert_to_masked.py --input-file ../resource/subsets/swe_doc_gen_locate_20.jsonl --output-file ../resource/swe_doc_gen_masked_20.jsonl
"""

import argparse
import ast
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

HOME_DIR = os.environ.get("HOME", "")
DOWNLOAD_DIR = os.path.join(HOME_DIR, "tmp/")


def clone_and_checkout(repo_name: str, commit: str, repo_dir: str) -> bool:
    """Clone repo if needed and checkout the given commit. Returns True on success."""
    try:
        if not os.path.exists(repo_dir):
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo_name}.git", repo_dir],
                check=True,
                capture_output=True,
            )
        subprocess.run(
            ["git", "-C", repo_dir, "fetch", "--all"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", repo_dir, "checkout", commit],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[warn] Failed to checkout {repo_name}@{commit}: {e}")
        return False


def remove_repo_dir(repo_dir: str) -> None:
    """Remove cloned repository directory to save disk space."""
    if os.path.exists(repo_dir):
        subprocess.run(["rm", "-rf", repo_dir], check=True)


def extract_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[Dict]:
    """Extract function signature details using AST."""
    parameters = []
    args = node.args
    defaults_offset = len(args.args) - len(args.defaults)

    for i, arg in enumerate(args.args):
        if arg.arg in ('self', 'cls'):
            continue

        param = {"name": arg.arg}

        if arg.annotation:
            try:
                param["type"] = ast.unparse(arg.annotation)
            except:
                param["type"] = None

        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            try:
                param["default"] = ast.unparse(args.defaults[default_idx])
            except:
                param["default"] = "..."

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
                pass
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

    return parameters


def extract_return_info(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[Dict]:
    """Extract return type annotation if present."""
    if node.returns:
        try:
            return {"type": ast.unparse(node.returns)}
        except:
            pass
    return None


def extract_function_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[Dict]:
    """Extract function calls made within the function body."""
    calls = []
    seen = set()

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, call_node):
            func_name = None
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id
            elif isinstance(call_node.func, ast.Attribute):
                try:
                    func_name = ast.unparse(call_node.func)
                except:
                    func_name = call_node.func.attr
            else:
                try:
                    func_name = ast.unparse(call_node.func)
                except:
                    pass

            if func_name and func_name not in seen:
                calls.append({"function": func_name})
                seen.add(func_name)

            self.generic_visit(call_node)

    visitor = CallVisitor()
    for child in node.body:
        visitor.visit(child)

    return calls[:10]  # Limit to 10


def extract_class_info(node: ast.ClassDef) -> Dict:
    """Extract class information including bases and methods."""
    info = {"bases": [], "methods": []}

    for base in node.bases:
        try:
            info["bases"].append(ast.unparse(base))
        except:
            pass

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info["methods"].append(item.name)

    return info


def get_ast_info_for_instance(instance: Dict, repo_dir: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Extract AST information for a given instance."""
    file_path = os.path.join(repo_dir, instance["file_path"])

    if not os.path.exists(file_path):
        return None, None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except (UnicodeDecodeError, IOError):
        return None, None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, None

    target_line = instance["module_line_start"]
    module_name = instance["module_name"]

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        node_start = node.lineno - 1  # Convert to 0-indexed
        if node_start != target_line:
            continue

        if node.name != module_name:
            continue

        # Found the target node
        detailed_info = {}

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            detailed_info["parameters"] = extract_function_signature(node)
            detailed_info["returns"] = extract_return_info(node)
            detailed_info["call_details"] = extract_function_calls(node)
        else:  # ClassDef
            class_info = extract_class_info(node)
            detailed_info["bases"] = class_info["bases"]
            detailed_info["methods"] = class_info["methods"]
            detailed_info["parameters"] = []
            detailed_info["returns"] = None
            detailed_info["call_details"] = []

        # Get source code
        source_lines = source.split('\n')
        module_end = instance["module_line_end"]
        func_source = '\n'.join(source_lines[target_line:module_end + 1])

        return detailed_info, func_source

    return None, None


def generate_masked_description_llm(
    docstring: str,
    module_name: str,
    module_type: str,
    detailed_info: Dict,
    source_code: str,
    model: str,
    temperature: float,
    use_anthropic: bool = False,
) -> Dict:
    """Use LLM to generate masked description with detailed info."""

    prompt = f"""You are preparing a masked docstring-location dataset. Your task is to analyze a Python {module_type} and generate structured information about it.

CRITICAL RULES:
1. NEVER mention the {module_type} name "{module_name}" in any description
2. Refer to it as "this {module_type}" or "the {module_type}" instead
3. Do not quote the docstring verbatim; summarize in your own words

Here is the {module_type} information:

Original docstring:
{docstring}

Source code (first 100 lines):
{source_code[:5000]}

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

    def parse_and_validate(response_text: str) -> Dict:
        """Parse JSON response and validate no name leakage."""
        # Try to extract JSON from response (handle markdown code blocks)
        json_text = response_text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        result = json.loads(json_text)

        # Validate that the name is not in the brief_description
        brief = result.get("brief_description", "")
        if module_name.lower() in brief.lower():
            brief = re.sub(
                rf'\b{re.escape(module_name)}\b',
                f"this {module_type}",
                brief,
                flags=re.IGNORECASE
            )
            result["brief_description"] = brief

        return result

    try:
        if use_anthropic:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"You are a helpful assistant that outputs valid JSON only.\n\n{prompt}"}
                ],
            )
            response_text = response.content[0].text.strip()
            return parse_and_validate(response_text)
        else:
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
            return parse_and_validate(response_text)

    except Exception as exc:
        print(f"[warn] LLM call failed ({exc}), using fallback")
        # Fallback: use docstring first line with name removed
        first_line = docstring.strip().split("\n")[0] if docstring else ""
        first_line = re.sub(
            rf'\b{re.escape(module_name)}\b',
            f"this {module_type}",
            first_line,
            flags=re.IGNORECASE
        )
        return {
            "brief_description": first_line,
            "parameters": detailed_info.get("parameters", []),
            "returns": detailed_info.get("returns"),
            "call_details": detailed_info.get("call_details", []),
        }


def convert_instance_to_masked(
    instance: Dict,
    repo_dir: str,
    model: str,
    temperature: float,
    use_anthropic: bool = False,
) -> Optional[Dict]:
    """Convert a single instance to masked format."""

    # Get AST info
    ast_info, source_code = get_ast_info_for_instance(instance, repo_dir)

    if ast_info is None:
        print(f"[warn] Could not extract AST info for {instance['instance_id']}")
        # Use empty defaults
        ast_info = {
            "parameters": [],
            "returns": None,
            "call_details": [],
        }
        source_code = ""

    # Generate masked description
    masked_info = generate_masked_description_llm(
        docstring=instance.get("docstring", ""),
        module_name=instance["module_name"],
        module_type=instance["module_type"],
        detailed_info=ast_info,
        source_code=source_code or "",
        model=model,
        temperature=temperature,
        use_anthropic=use_anthropic,
    )

    # Build new instance
    new_instance = {
        # Core instance info (preserved from original)
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "image_instance_id": instance.get("image_instance_id", instance["instance_id"].rsplit("_", 1)[0]),

        # File and location info
        "file_path": instance["file_path"],
        "module_name": instance["module_name"],  # Stored but NOT used in prompts
        "module_type": instance["module_type"],
        "module_line_start": instance["module_line_start"],
        "module_line_end": instance["module_line_end"],

        # Docstring info
        "docstring": instance.get("docstring", ""),
        "docstring_line_start": instance["docstring_line_start"],
        "docstring_line_end": instance["docstring_line_end"],

        # NEW: Masked description (does NOT contain function name)
        "brief_description": masked_info.get("brief_description", ""),

        # NEW: Detailed information for tunable difficulty
        "parameters": masked_info.get("parameters", []),
        "returns": masked_info.get("returns"),
        "call_details": masked_info.get("call_details", []),

        # Raw AST info
        "ast_info": ast_info,

        # DEPRECATED: Keep original for comparison (but don't use in prompts)
        "_original_function_description": instance.get("function_description", ""),
    }

    return new_instance


def main():
    parser = argparse.ArgumentParser(
        description="Convert existing swe_doc_gen_locate data to masked format"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input JSONL file with existing data",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output JSONL file for masked data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model used to synthesize descriptions (OpenAI or Anthropic model name)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--use-anthropic",
        action="store_true",
        help="Use Anthropic API instead of OpenAI",
    )
    parser.add_argument(
        "--keep-repos",
        action="store_true",
        help="Don't delete repos after processing",
    )
    args = parser.parse_args()

    # Set default model based on API choice
    if args.use_anthropic and args.model == "gpt-4o-mini":
        args.model = "claude-3-5-haiku-20241022"  # Default Anthropic model

    print(f"[info] Using {'Anthropic' if args.use_anthropic else 'OpenAI'} API with model: {args.model}")

    # Load input data
    instances = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    print(f"[info] Loaded {len(instances)} instances from {args.input_file}")

    # Group by repo to minimize cloning
    repo_instances = {}
    for inst in instances:
        repo = inst["repo"]
        if repo not in repo_instances:
            repo_instances[repo] = []
        repo_instances[repo].append(inst)

    print(f"[info] Found {len(repo_instances)} unique repos")

    # Process each repo
    results = []
    failed = []

    for repo, repo_insts in tqdm(repo_instances.items(), desc="Processing repos"):
        repo_dir = os.path.join(DOWNLOAD_DIR, repo.split("/")[-1])

        # Get commit (use first instance's commit)
        commit = repo_insts[0]["base_commit"]

        # Clone and checkout
        if not clone_and_checkout(repo, commit, repo_dir):
            print(f"[warn] Skipping repo {repo}")
            failed.extend([inst["instance_id"] for inst in repo_insts])
            continue

        # Process each instance in this repo
        for inst in tqdm(repo_insts, desc=f"{repo}", leave=False):
            new_inst = convert_instance_to_masked(
                instance=inst,
                repo_dir=repo_dir,
                model=args.model,
                temperature=args.temperature,
                use_anthropic=args.use_anthropic,
            )
            if new_inst:
                results.append(new_inst)
            else:
                failed.append(inst["instance_id"])

        # Clean up
        if not args.keep_repos:
            remove_repo_dir(repo_dir)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n[info] Saved {len(results)} masked instances to {args.output_file}")
    if failed:
        print(f"[warn] Failed to process {len(failed)} instances: {failed}")

    # Print sample output
    if results:
        print("\n[info] Sample output:")
        sample = results[0]
        print(f"  instance_id: {sample['instance_id']}")
        print(f"  module_name: {sample['module_name']} (stored but not in description)")
        print(f"  brief_description: {sample['brief_description'][:200]}...")
        print(f"  parameters: {json.dumps(sample['parameters'][:2], indent=2)}")
        print(f"  returns: {json.dumps(sample['returns'], indent=2)}")
        print(f"  call_details: {json.dumps(sample['call_details'][:3], indent=2)}")


if __name__ == "__main__":
    main()
