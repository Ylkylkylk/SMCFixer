# SMCFixer
## Introduction
Solidity, the dominant smart contract language for
Ethereum, has rapidly evolved with frequent version updates
to enhance security, functionality, and developer experience.
However, these continual changes introduce significant chal-
lenges, particularly in compilation errors, code migration, and
maintenance, making it increasingly difficult for developers to
adapt. Our study systematically investigates these challenges by
analyzing the impact of Solidity version evolution, revealing that
81.69% of examined contracts encounter errors when compiled
across different versions, with 86.92% of compilation errors.

To address these issues, we evaluated the effectiveness of large
language models (LLMs) in resolving Solidity compilation errors
caused by version migrations. Through extensive experiments
on open-source LLMs (GPT-4o, GPT-3.5-turbo) and closed-
source (LLaMa3, DeepSeek) LLMs, we find that while LLMs
demonstrate potential to fix errors, their performance varies
significantly depending on the type of error and prompt granu-
larity. Our findings highlight the importance of domain-specific
knowledge in improving LLM-driven solutions for Solidity repair.

Based on these insights, we propose SMCFIXER, a novel
framework that integrates expert knowledge retrieval and LLM-
based repair mechanisms to enhance Solidity compilation error
resolution. SMCFIXER consists of three core components: code
slicing, knowledge retrieval, and patch generation, designed to
extract relevant error information, retrieve expert knowledge
from official documentation, and iteratively generate patches for
Solidity migration. Experimental results show that our approach
significantly improves repair accuracy across various Solidity
versions, achieving a 24.24% improvement over standalone GPT-
4o on real-world datasets, with a peak accuracy of 96.97%

## Data
SMCFIXER is designed for evaluating the ability of large language models (LLMs) to fix Solidity compilation errors caused by version migrations. It provides three datasets:

### 📊 Statistics
- **DATASET-A**: Automatically generated dataset based on official Solidity breaking changes.
- **DATASET-B**: Benchmark dataset for evaluating patch generation performance.
- **DATASET-C**: Real-world dataset collected from open-source platforms.

Each dataset contains Solidity code instances with version-specific compilation errors and their corresponding fixed versions. These datasets enable both correctness and robustness evaluation for LLM-based repair systems.

| Dataset    | #Instance | #BreakingChange | #ErrorTypes                              |
|------------|-----------|------------------|-------------------------------------------|
| DATASET-A  | 2050      | 131              | Parser, Declaration, Syntax, Type         |
| DATASET-B  | 1460      | 93               | Parser, Declaration, Syntax, Type         |
| DATASET-C  | 33        | —                | Real-world Solidity compilation failures  |

### 📂 File Descriptions
- `Dataset/dataset_A/`:  
  Automatically generated with GPT-4 based on 131 documented breaking changes in Solidity. Used to train and evaluate the retriever and generation components.

- `Dataset/dataset_B/`:  
  A curated subset of DATASET-A, focusing on 93 compilation-related breaking changes. Used in model comparison and ablation studies.

- `Dataset/dataset_C/`:  
  Real-world smart contracts from GitHub, OpenZeppelin, Reddit, and Stack Overflow. Each contract fails to compile due to version migration issues and requires repair.


## Usage
First install the required Python packages

    pip install -r requirements.txt

Install Solidity compiler `solcjs`

    npm install -g solc

You can use SMCFIXER with the following command-line instruction：

    python SMCFixer_run.py --file <path_to_solidity_file> [options]

| options   | Description |
| ------ | ---- | 
| `--top1`   | Returns the top-1 relevent knowledge retrieved by SocR   |
| `--top3`   | Returns the top-3 relevent knowledge retrieved by SocR   |
| `--top5`   | Returns the top-5 relevent knowledge retrieved by SocR   |

Example:

    python SMCFixer_run.py --file contracts/MyContract.sol --top1




