# iSHAP: Interpretable SHAP for Machine Learning Models

iSHAP is a Python library designed to enhance the interpretability of machine learning models by providing more interpretable SHAP (SHapley Additive exPlanations) values. This tool is particularly useful for understanding complex models by offering clear insights into feature importance and model predictions.

## Features

- **Enhanced Interpretability**: Provides more interpretable SHAP values for complex models.
- **Synthetic and Real-World Data Support**: Includes experiments for both synthetic datasets and real-world scenarios.
- **User-Friendly Interface**: Designed for ease of use, facilitating quick integration into existing workflows.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Schascha1/iSHAP.git
```

Navigate to the project directory:

```bash
cd iSHAP
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Notebook
We provide a [notebook](example_usage.ipynb) that showcases how to run and visualize iSHAP on a common regression. 

### Running Synthetic Experiments

To execute the synthetic experiments as described in Section 5.1 of the associated paper, run:

```bash
bash run_synthetic_experiments.sh
```

### Running Real-World Experiments

For the real-world experiments detailed in Section 5.2 of the paper, execute:

```bash
bash run_interpolation_experiments.sh
```

## Repository Structure

- `data/`: Contains datasets used for experiments.
- `results/`: Stores the output and results from experiments.
- `scripts/`: Includes shell scripts to run experiments.
- `src/`: Source code for the iSHAP library.
- `notebooks/`: Jupyter notebooks demonstrating usage and experiments.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Schascha1/iSHAP/blob/main/LICENSE) file for details.

## Citation

If you use iSHAP in your research, please cite the associated paper:

```bibtex
@inproceedings{ishap:xu:25,
  title={Succint Interaction-Aware Explanations},
  author={Xu, Sascha and C{\"u}ppers, Joscha and Vreeken, Jilles},
  booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025},
  url={https://openreview.net/forum?id=uXLXFWTaoT}
}
```

## Contact

For questions or support, please open an issue in the repository or contact [sascha.xu@cispa.de](mailto:sascha.xu@cispa.de).

---

This README provides an overview of the iSHAP project, its installation and usage. For detailed information, refer to the [associated paper](https://eda.rg.cispa.io/prj/ishap/). 