# Survey-CodeLLM4LowResource-DSL
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2410.03981-b31b1b.svg)](https://arxiv.org/pdf/2410.03981)
[![GitHub Stars](https://img.shields.io/github/stars/username/Survey-CodeLLM4LowResource-DSL?style=social)](https://github.com/username/Survey-CodeLLM4LowResource-DSL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-September%202025-blue)]()

</div>

This repository contains resources referenced in the paper: [A Survey on LLM-based Code Generation for Low-Resource and Domain-Specific Programming Languages](https://arxiv.org/pdf/2410.03981).



---

## 🔥 What's New

- **[September 2025]** 📄 Our survey paper is accepted at ACM Transactions on Software Engineering and Methodology
(TOSEM)
- **[September 2025]** 📊 Comprehensive analysis of 111 papers covering 40+ programming languages
- **[September 2025]** 🚀 Repository launched with complete paper categorization and analysis

## 📖 Overview

Large Language Models (LLMs) have shown remarkable capabilities in code generation for popular programming languages. However, their performance in **Low-Resource Programming Languages (LRPLs)** and **Domain-Specific Languages (DSLs)** remains a critical challenge. This gap affects millions of developers - with Rust alone having 3.5 million users - who are currently unable to fully leverage LLM capabilities. 

Our survey fills this gap by providing a **systematic review of 111 papers** filtered from over 27,000 published studies from 2020-2025, investigating the capabilities and limitations of LLMs in these specialized domains. We identify four main evaluation techniques, categorize enhancement methods into six groups, and analyze dataset curation approaches for LRPLs and DSLs.

<div align="center">

<img src="assets/heatmap.PNG" width="800px">

*Figure 3: Performance comparison between High-Resource Programming Languages (HRPLs) and Low-Resource Programming Languages (LRPLs) on the MultiPL-E benchmark. The heatmap shows significant performance disparities, between for example, Python and Rust.*

</div>

### 🎯 Key Contributions

- **Comprehensive Literature Review**: Systematic analysis of 111 papers from 27,000+ studies (2020-2025)
- **Performance Gap Analysis**: Quantitative comparison showing significant disparities between HRPLs and LRPLs/DSLs
- **Methodology Taxonomy**: Six categories of enhancement techniques with effectiveness analysis
- **Evaluation Framework**: Four evaluation approaches with domain-specific metrics
- **Dataset Analysis**: Comprehensive review of data curation and preparation strategies
- **Research Roadmap**: Identification of challenges and opportunities for future research


## 📋 Complete Paper List

### Low-Resource Programming Languages (LRPLs) - 51 Papers

#### Rust (8 papers)
- [128] **OctoPack: Instruction tuning code large language models** - *Muennighoff et al., 2023* - NeurIPS Workshop
- [30] **Assessing Code Generation with Intermediate Languages** - *Deng et al., 2024* - arXiv
- [47] **ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation** - *Ren et al., 2024* - arXiv  
- [187] **MagiCoder: Empowering Code Generation with OSS-INSTRUCT** - *Wei et al., 2024* - ICML
- [140] **IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators** - *Paul et al., 2024* - ACL
- [176] **Investigating the performance of language models for completing code in functional programming languages: a haskell case study** - *Van Dam et al., 2024* - ICSE
- [15] **McEval: Massively Multilingual Code Evaluation** - *Chai et al., 2025* - ICLR
- [64] **Kotlin ML Pack: Technical Report** - *Titov et al., 2024* - arXiv

#### R (6 papers)
- [13] **Knowledge transfer from high-resource to low-resource programming languages for code llms** - *Cassano et al., 2024* - OOPSLA
- [14] **MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation** - *Cassano et al., 2023* - TSE
- [33] **GenCodeSearchNet: A Benchmark Test Suite for Evaluating Generalization in Programming Language Understanding** - *Diera et al., 2023* - GenBench Workshop
- [125] **User Centric Evaluation of Code Generation Tools** - *Miah & Zhu, 2024* - AITest
- [147] **Time-Efficient Code Completion Model for the R Programming Language** - *Popov et al., 2021* - NLP4Prog Workshop
- [118] **Python is Not Always the Best Choice: Embracing Multilingual Program of Thoughts** - *Luo et al., 2024* - EMNLP

#### Kotlin (4 papers)
- [64] **Kotlin ML Pack: Technical Report** - *Titov et al., 2024* - arXiv
- [6] **Multi-lingual evaluation of code generation models** - *Athiwaratkun et al., 2023* - Amazon Science
- [102] **XCodeEval: An Execution-based Large Scale Multilingual Multitask Benchmark** - *Khan et al., 2024* - ACL
- [92] **Language models for code completion: A practical evaluation** - *Izadi et al., 2024* - ICSE

#### Bash/Shell (5 papers)
- [146] **DocCGen: Document-based Controlled Code Generation** - *Pimparkhede et al., 2024* - EMNLP
- [162] **ShellGPT: Generative Pre-trained Transformer Model for Shell Language Understanding** - *Shi et al., 2023* - ISSRE
- [179] **Tackling Execution-Based Evaluation for NL2Bash** - *Vo et al., 2024* - arXiv
- [196] **InterCode: standardizing and benchmarking interactive coding with execution feedback** - *Yang et al., 2023* - NeurIPS
- [215] **DocPrompting: Generating Code by Retrieving the Docs** - *Zhou et al., 2023* - ICLR

#### Multi-Language Studies (15+ papers)
- [137] **Measuring the impact of programming language distribution** - *Orlanski et al., 2023* - ICML
- [188] **Batched Low-Rank Adaptation of Foundation Models** - *Wen & Chaudhuri, 2024* - ICLR
- [193] **CodeScope: An Execution-based Multilingual Multitask Multidimensional Benchmark** - *Yan et al., 2024* - ACL
- [143] **HumanEval-XL: A Multilingual Code Generation Benchmark** - *Peng et al., 2024* - LREC-COLING
- [158] **StackEval: Benchmarking llms in coding assistance** - *Shah et al., 2024* - NeurIPS

### Domain-Specific Languages (DSLs) - 59 Papers

#### Hardware Description Languages (25 papers)

**Verilog (20 papers):**
- [19] **Data is all you need: Finetuning llms for chip design via an automated design-data augmentation framework** - *Chang et al., 2024* - DAC
- [25] **Origen: Enhancing rtl code generation with code-to-code augmentation and self-reflection** - *Cui et al., 2024* - ICCAD
- [78] **Autovcoder: A systematic framework for automated verilog code generation using llms** - *Gao et al., 2024* - ICCD
- [80] **From English to ASIC: Hardware Implementation with Large Language Model** - *Goh et al., 2024* - arXiv
- [113] **Verilogeval: Evaluating large language models for verilog code generation** - *Liu et al., 2023* - ICCAD
- [114] **RTLCoder: Outperforming GPT-3.5 in Design RTL Generation** - *Liu et al., 2024* - LAD Workshop
- [117] **Rtllm: An open-source benchmark for design rtl generation with large language model** - *Lu et al., 2024* - ASP-DAC
- [130] **A multi-expert large language model architecture for verilog code generation** - *Nadimi & Zheng, 2024* - LAD Workshop
- [142] **BetterV: controlled verilog generation with discriminative guidance** - *Pei et al., 2024* - ICML
- [149] **AutoBench: Automatic Testbench Generation and Evaluation Using LLMs for HDL Design** - *Qiu et al., 2024* - MLCAD
- [170] **Benchmarking large language models for automated verilog rtl code generation** - *Thakur et al., 2023* - DATE
- [172] **Advanced Large Language Model (LLM)-Driven Verilog Development** - *Thorat et al., 2023* - arXiv
- [177] **VHDLEval: A Framework for Evaluating Large Language Models in VHDL Code Generation** - *Vijayaraghavan et al., 2024* - LAD Workshop
- [208] **MG-Verilog: Multi-grained Dataset Towards Enhanced LLM-assisted Verilog Generation** - *Zhang et al., 2024* - LAD Workshop
- [210] **CodeV: Empowering LLMs for Verilog Generation through Multi-Level Summarization** - *Zhao et al., 2024* - arXiv

#### Infrastructure & Automation (7 papers)
- [132] **KubePlaybook: A Repository of Ansible Playbooks for Kubernetes Auto-Remediation with LLMs** - *Namrud et al., 2024* - ICPE
- [146] **DocCGen: Document-based Controlled Code Generation** - *Pimparkhede et al., 2024* - EMNLP
- [148] **Automated Code Generation for Information Technology Tasks in YAML through Large Language Models** - *Pujar et al., 2025* - DAC
- [154] **Ansible lightspeed: A code generation service for it automation** - *Sahoo et al., 2024* - ASE
- [207] **On the effectiveness of large language models for github workflows** - *Zhang et al., 2024* - ARES
- [103] **Iac-eval: A code generation benchmark for cloud infrastructure-as-code programs** - *Kon et al., 2024* - NeurIPS

#### Formal Methods & Verification (10 papers)
- [16] **Towards Neural Synthesis for SMT-Assisted Proof-Oriented Programming** - *Chakraborty et al., 2025* - ICSE
- [36] **Towards a Mathematics Formalisation Assistant using Large Language Models** - *Agrawal et al., 2022* - arXiv
- [41] **FIMO: A Challenge Formal Dataset for Automated Theorem Proving** - *Liu et al., 2023* - arXiv
- [74] **Enhancing Formal Theorem Proving: A Comprehensive Dataset for Training AI Models on Coq Code** - *Florath, 2024* - arXiv
- [191] **Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data** - *Xin et al., 2024* - NeurIPS Workshop
- [200] **Leveraging Large Language Models for Automated Proof Synthesis in Rust** - *Yao et al., 2023* - arXiv
- [126] **Synthetic programming elicitation for text-to-code in very low-resource programming and formal languages** - *Mora et al., 2024* - NeurIPS

#### Logic & Specification Languages (8 papers)
- [24] **nl2spec: Interactively translating unstructured natural language to temporal logics with large language models** - *Cosler et al., 2023* - CAV
- [85] **Formal Specifications from Natural Language** - *Hahn et al., 2022* - arXiv
- [136] **LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers** - *Olausson et al., 2023* - EMNLP
- [199] **Harnessing the Power of Large Language Models for Natural Language to First-Order Logic Translation** - *Yang et al., 2024* - ACL

#### Scientific & Specialized Domains (9 papers)
- [97] **Flame: A small language model for spreadsheet formulas** - *Joshi et al., 2024* - AAAI
- [139] **Sketchgen: Generating constrained cad sketches** - *Para et al., 2021* - NeurIPS
- [145] **What If: Generating Code to Answer Simulation Questions in Chemistry Texts** - *Peretz et al., 2023* - SIGIR
- [165] **Errors are Useful Prompts: Instruction Guided Task Programming with Verifier-Assisted Iterative Prompting** - *Skreta et al., 2023* - arXiv
- [166] **Generating consistent PDDL domains with Large Language Models** - *Smirnov et al., 2024* - arXiv
- [180] **Grammar prompting for domain-specific language generation with large language models** - *Wang et al., 2023* - NeurIPS

---

## 🏆 Major Benchmarks & Datasets

### Multi-Language Benchmarks

| Benchmark | Languages | Papers | Description |
|-----------|-----------|--------|-------------|
| **MultiPL-E** | Bash, Lua, Perl, R, Ruby, Racket, D, Go, Julia, Rust, Scala, Swift | [13], [14], [57], [137], [140], [187], [188], [190], [205] | First massive multi-lingual benchmark including LRPLs |
| **xCodeEval** | Kotlin, Ruby, Rust (+ 8 others) | [102] | 2.5K problems with executable evaluation |
| **CodeScope** | Ruby, Kotlin, D, Perl, Rust, Delphi (+ 8 others) | [193] | Multi-dimensional evaluation framework |
| **MBXP** | Ruby, Kotlin, Scala, Swift, Perl | [6] | Mathematical reasoning benchmark |
| **BabelCode** | Dart, Lua, Rust, C#, R, Julia, Haskell | [137] | Language diversity and syntax transfer |
| **MCEval** | 40 programming languages | [15] | 16K samples across diverse languages |

### Domain-Specific Benchmarks

#### Hardware Design
| Benchmark | Language | Papers | Description |
|-----------|----------|--------|-------------|
| **VerilogEval** | Verilog | [25], [78], [80], [113], [114], [130], [142], [149], [172], [177], [208], [210] | Standard Verilog benchmark (widely adopted) |
| **RTLLM** | Verilog | [19], [25], [78], [114], [117], [172], [210] | 30 digital designs with testbenches |
| **VHDL-Eval** | VHDL | [177] | VHDL code generation for FPGA development |

#### Infrastructure & Automation
| Benchmark | Language | Papers | Description |
|-----------|----------|--------|-------------|
| **TLDR** | Bash | [146], [215] | 1,879 Bash commands from community project |
| **InterCode** | Bash | [196] | Interactive coding environment |
| **NL-to-Ansible** | Ansible | [146] | Infrastructure automation tasks |
| **IaC-Eval** | Terraform HCL | [103] | Cloud Infrastructure-as-Code generation |

#### Formal Methods
| Benchmark | Language | Papers | Description |
|-----------|----------|--------|-------------|
| **FIMO** | Lean | [41], [191] | IMO theorem proving challenges |
| **miniF2F** | Lean | [41], [191] | Mathematical theorem proving |
| **FOLIO** | FOL | [136], [199] | Natural language to logic reasoning |

---

## 🛠️ Enhancement Techniques Analysis

### Model Adaptation Techniques

#### Fine-tuning Success Stories
| Domain | Model | Base | Fine-tuned | Improvement | Paper |
|--------|-------|------|------------|-------------|-------|
| **Verilog** | CodeQwen | 60.0% | 68.1% | +8.1% | [142] |
| **Verilog** | CodeLlama | 60.0% | 78.1% | +18.1% | [210] |
| **Lean** | DeepSeek | 27.5% | 52.0% | +24.5% | [191] |
| **Rust** | CodeLLaMA | 27.0% | 40.3% | +13.3% | [187] |
| **Haskell** | CodeGPT | 23.2% | 40.0% | +16.8% | [176] |

#### Most Popular Base Models
| Model Family | Usage Count | Key Papers |
|--------------|-------------|------------|
| **LLaMA Family** | 14 papers | [71], [140], [187], [199], [210] |
| **DeepSeek Family** | 10 papers | [25], [78], [84], [140], [142] |
| **StarCoder Family** | 9 papers | [13], [62], [188] |
| **CodeGen** | 6 papers | [134] |
| **T5/CodeT5** | 4 papers | [85], [109], [184] |

### Prompting Strategies

#### Advanced Prompting Techniques
| Technique | Papers | Languages | Key Benefits |
|-----------|--------|-----------|--------------|
| **Few-shot** | [1], [45], [170], [174], [177] | OCL, Vega-lite, Verilog, DSL, VHDL | Better accuracy than zero-shot |
| **Chain-of-Thought** | [199] | FOL | Step-by-step logical reasoning |
| **Grammar Prompting** | [180] | SMILES, PDDL, GeoQuery | Domain-specific syntax guidance |
| **Hierarchical** | [131], [122] | Verilog | Complex task decomposition |
| **Self-Planning** | [117] | Verilog | Model generates plan before code |

### Iterative Feedback Approaches

| Domain | Tool Used | Papers | Feedback Type |
|--------|-----------|--------|---------------|
| **Verilog** | iVerilog | [10], [122], [172] | Compilation errors |
| **PLC** | MATIEC, NuXmv | [71] | Grammar + formal verification |
| **Chemistry** | Rule-based verifier | [165] | Syntax checking |
| **Rust/Verus** | Verus verifier | [200] | Proof validation |
| **PDDL** | FastDownward planner | [166] | Planning consistency |

---

## 📊 Evaluation Metrics & Approaches

### Automatic Evaluation Metrics

#### Functional Correctness
| Metric | Languages | Papers | Description |
|--------|-----------|--------|-------------|
| **pass@k** | 35+ languages | 30+ papers | Most common metric across LRPLs/DSLs |
| **Execution Accuracy** | Bash, CQL | [53], [179] | Real execution in controlled environments |
| **Compilation Rate** | Verilog, SVA | [99] | Syntactic correctness validation |

#### Similarity Metrics
| Metric | Languages | Usage | Description |
|--------|-----------|-------|-------------|
| **BLEU** | 20+ languages | 12 papers | Text similarity comparison |
| **ROUGE** | 16+ languages | 5 papers | Sequence matching evaluation |
| **Edit Similarity** | 6 languages | 4 papers | Character/token level differences |
| **Exact Match** | 15+ languages | 13 papers | Binary correctness measure |

### Domain-Specific Evaluation

#### Hardware Design (Verilog/VHDL)
| Metric | Description | Papers |
|--------|-------------|--------|
| **Power-Performance-Area (PPA)** | Multi-dimensional hardware evaluation | [50], [117], [122], [172] |
| **Syn-VCS, Syn-DC** | Synthesis-based correctness | [114] |
| **Area-Delay Product** | Physical implementation metrics | [56] |

#### Logic & Formal Languages
| Metric | Description | Papers |
|--------|-------------|--------|
| **Semantic Accuracy** | Logical equivalence for Regex/FOL | [85], [115] |
| **Logical Equivalence** | Truth table comparison | [199] |
| **Verify@k** | Formal verification success rate | [16] |

#### Infrastructure Languages
| Metric | Description | Papers |
|--------|-------------|--------|
| **Ansible Aware Metric** | Domain-specific syntax evaluation | [146], [148] |
| **Schema Correct Metric** | Structural validation | [146], [148] |
| **Command Accuracy** | Shell command correctness | [146], [215] |

### Manual Evaluation Approaches

#### Expert Assessment Requirements
| Domain | Expert Type | Evaluation Focus | Papers |
|--------|-------------|------------------|--------|
| **Hardware** | Hardware engineers | Design quality, synthesis | [10], [99], [121] |
| **Chemistry** | Chemists | Scientific validity | [165] |
| **Mathematics** | Mathematicians | Proof correctness | [74], [191] |
| **PLC** | Control engineers | Industry standards | [71] |

---

## 📚 Dataset Curation & Processing

### Data Sources by Category

#### Code Repositories
| Source | Languages | Papers | Description |
|--------|-----------|--------|-------------|
| **GitHub** | Most LRPLs/DSLs | 25+ papers | Primary source for code data |
| **GitLab** | Various | [148] | Alternative repository platform |
| **Programming Contests** | Multiple | [102], [140] | Clean, self-contained problems |

#### Educational Resources
| Source | Languages | Papers | Usage |
|--------|-----------|--------|-------|
| **Textbooks** | R, Verilog | [125], [170] | High-quality examples |
| **HDLBits** | Verilog | [113] | Hardware design exercises |
| **University Courses** | Lean, Coq | [36], [191] | Formal methods materials |

#### Specialized Sources
| Source | Domain | Papers | Content Type |
|--------|--------|--------|--------------|
| **Stack Overflow** | Multiple | [28], [57] | Q&A and problem-solving |
| **Technical Forums** | Hardware, Excel | [97], [99] | Community knowledge |
| **Industrial Datasets** | Ansible, PLC | [154], [71] | Real-world applications |

### Synthetic Data Generation

#### LLM-based Generation
| Generator Model | Target Languages | Papers | Approach |
|-----------------|------------------|--------|----------|
| **GPT-3.5/4** | Verilog, Kotlin, FOL | [113], [64], [199] | Problem-solution generation |
| **Claude-3-Haiku** | Verilog | [25] | Code-to-code augmentation |
| **StarCoder-15B** | Multiple LRPLs | [13] | Cross-language translation |
| **DeepSeek** | Lean | [191] | Quality classification |

### Data Processing Pipeline

#### Standard Processing Steps
1. **Initial Filtering**: File extensions, size limits, license compatibility
2. **Deduplication**: MinHash, ROUGE-L based approaches  
3. **Fine-grained Filtering**: Syntax validation, domain rules
4. **Code Extraction**: Comment separation, structure preservation
5. **Quality Checks**: Compilation tests, static analysis
6. **Dataset-specific Processing**: Domain adaptations
7. **Decontamination**: Benchmark data removal

#### Domain-Specific Adaptations
| Domain | Special Processing | Papers | Tools Used |
|--------|-------------------|--------|------------|
| **Verilog** | Module extraction, synthesis validation | [113], [210] | PyVerilog, iVerilog |
| **R** | RMD to R conversion, metadata addition | [33], [147] | Custom parsers |
| **Excel** | Formula normalization, case handling | [97] | Custom tokenizers |
| **Hardware** | Visual element conversion to text | [113] | Manual annotation |
 
## 📞 Contact
- **Sathvik Joel**
  - 📧 [ksjoe30@gmail.com](mailto:ksjoe30@gmail.com)

- **Jie JW Wu** (Repo Maintainer)
  - 📧 [jie.jw.wu@ubc.ca](mailto:jie.jw.wu@acm.org)

- **Fatemeh Fard** 
  - 📧 [fatemeh.fard@ubc.ca](mailto:fatemeh.fard@ubc.ca)  


## 📜 Citation  

```bibtex
@article{joel2024survey,
  title={A survey on llm-based code generation for low-resource and domain-specific programming languages},
  author={Joel, Sathvik and Wu, Jie JW and Fard, Fatemeh H},
  journal={arXiv preprint arXiv:2410.03981},
  year={2024}
}
```


<div align="center">

**🌟 If you find this survey useful, please give it a star! 🌟**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=username.Survey-CodeLLM4LowResource-DSL)

---

*Last updated: Septemper 2025*

</div>