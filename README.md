<a name="readme-top"></a>

# 🎨Hybrid-Gym: Training Coding Agents to Generalize Across Tasks

<p align="left">
  <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://arxiv.org/abs"><img src="https://img.shields.io/badge/arXiv--b31b1b.svg"></a>
</p>

Code for "Hybrid-Gym: Training Coding Agents to Generalize Across Tasks".
<br><br>
<img width="1004" height="357" alt="image" src="https://github.com/user-attachments/assets/37c9af75-4c23-4913-b74f-c1d910df2e11" />
<br><br>

## Overview

This repository is forked from [OpenHands](https://github.com/All-Hands-AI/OpenHands) at commit [`7bea93b1`](https://github.com/yiqingxyq/OpenHands2/tree/7bea93b1b668dbd89a447ea245f76be466a88ef7/evaluation/benchmarks). 
It contains four Hybrid Gym evaluation benchmarks for evaluating AI coding agents on targeted software engineering sub-tasks.

Our datasets and models are uploaded to [HuggingFace](https://huggingface.co/hybrid-gym).

<br><br>

## Benchmark Tasks

Please refer to the `README.md` file for each task for details.

| Task | Description | Dataset | README |
|------|-------------|---------|--------|
| **Function Generation** | Implement a function body given its signature and docstring | [hybrid-gym/hybrid_gym_func_gen](https://huggingface.co/datasets/hybrid-gym/hybrid_gym_func_gen) | [README](evaluation/benchmarks/hybrid_gym_func_gen/README.md) |
| **Function Localization** | Locate a function by its description and add a docstring | [hybrid-gym/hybrid_gym_func_localize](https://huggingface.co/datasets/hybrid-gym/hybrid_gym_func_localize) | [README](evaluation/benchmarks/hybrid_gym_func_localize/README.md) |
| **Dependency Search** | Find all functions/classes called by a target function and annotate them | [hybrid-gym/hybrid_gym_dep_search](https://huggingface.co/datasets/hybrid-gym/hybrid_gym_dep_search) | [README](evaluation/benchmarks/hybrid_gym_dep_search/README.md) |
| **Issue Localization** | Locate code related to a GitHub issue and add comments | [SWE-Gym/SWE-Gym-Raw](https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Raw) | [README](evaluation/benchmarks/hybrid_gym_issue_localize/README.md) |

<br><br>

## Convert Successful Trajectories to Training Data
After obtaining a `.jsonl` file of successful trajectories, we first compress it by running:
```
poetry run python evaluation/combine_final_completions.py $SUCCESS_OUTPUT_FILE
```

Then we convert it to the format of multi-turn conversations for training:
```
python evaluation/convert_data.py $COMPRESSED_SUCCESS_OUTPUT_FILE
```

The returned file is a `.jsonl` file, where each line is a training trajectory.

<br><br>

## Setup

Please refer to the [OpenHands documentation](https://docs.all-hands.dev/usage/getting-started) for setting up your local development environment and LLM configuration.

<br><br>

## Cite

If you find our paper or code useful, please cite the paper:

```
@misc{hybrid-gym,
      title={Hybrid-Gym: Training Coding Agents to Generalize Across Tasks}, 
      author={Yiqing Xie and Emmy Liu and Gaokai Zhang and Nachiket Kotalwar and Shubham Gandhi and Sathwik Acharya and Xingyao Wang and Carolyn Rose and Graham Neubig and Daniel Fried},
      year={2026},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
}
```

<br><br>

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.


