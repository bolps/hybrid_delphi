# Hybrid Delphi

Hybrid Delphi is a Python framework for conducting hybrid Delphi studies, integrating artificial intelligence and human participation for the evaluation and validation of rubrics and decision protocols. The project facilitates the collection, analysis, and synthesis of expert judgments, leveraging advanced language models and customizable workflows.

## Scientific Reference
This repository accompanies the paper:
**"Hybrid Delphi: A framework for collaborative validation with AI and human experts"**

> For methodological details, see the attached draft: `2026___CHI_Hybrid_Delphi-2.pdf`.

## Project Structure
- `delphi_consensus_pipeline_3round_rubric_graph.py`: main script for the 3-round Delphi pipeline.
- `logs/`: contains logs of Delphi sessions, organized by role, AI model, and round.
- `methodological notes/`: methodological notes and validation protocols.
- `results/`: outputs and analysis results.
- `archive.zip`: data archive or previous versions.
- `pyproject.toml`, `uv.lock`: Python dependency management.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bolps/hybrid_delphi.git
   cd hybrid_delphi
   ```
2. Install dependencies (recommended: [uv](https://github.com/astral-sh/uv)):
   ```bash
   uv pip install -r pyproject.toml
   ```
   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main pipeline:
```bash
python delphi_consensus_pipeline_3round_rubric_graph.py
```
Session logs will be saved in the `logs/` folder.

## Methodological Notes
See the `methodological notes/` folder for details on:
- Hybrid validation protocol
- Handling dimensions and rubrics
- Best practices for AI-human integration

## Credits
- Author: bolps
- For questions or collaborations, open an issue on GitHub.

## License
This project is released under the MIT license.
