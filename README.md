# atlas


[![rileyhickman](https://circleci.com/gh/rileyhickman/atlas.svg?style=svg&circle-token=96039a8d33f9fade7e4c1a5420312b0711b16cde)](https://app.circleci.com/pipelines/github/rileyhickman/atlas)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`atlas` is a Python package for Bayesian optimization in the experimental science. At its core, the package provides high-performing, easy-to-use Bayesian optimization based
on Gaussian processes (with help from the GPyTorch and BoTorch libraries). `atlas` attempts to cater directly to the needs of researchers in the experimental sciences,
and provides additional optimization capabilities to tackle problems typically encountered in such disciplines. These capabilities include optimization of categorical, discrete, and mixed parameter
spaces, multi-objective optimization, noisy optimization (noise on input parameters and objective measurements), constrained optimization (known and unknown constraints on the parameter space), multi-fidelity
optimization, meta-learning optimization, data-driven search space expansion/contraction, and more!

`atlas` is intended serve as the brain or experiment planner for self-driving laboratories.


`atlas` is developed at the University of Toronto, the Vector Institute for Artificial Intelligence, and Soteria Therapeutics, Inc.


![alt text](https://github.com/rileyhickman/atlas/blob/main/static/atlas_logo.png)

## Installation

Install `atlas` from source by executing the following commands

```bash
git clone git@github.com:rileyhickman/atlas.git
cd atlas
pip install -e .
```

To use the Google doc feature, you must install the Google client library

```
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

and `gspread`

```
pip install gspread
```

## Usage


### Proof-of-concept optimization


### Optimization of mixed-parameter spaces


### Optimization with a-priori known constraints


### Optimization with a-priori unknown constraints

### Multi-objective optimization

### Robust optimization with Golem


## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/)
 license. See `LICENSE` for more information.

## Contact

Academic collaborations and extensions/improvements to the code by the community
are encouraged. Please reach out to [Riley](riley.hickman@mail.utoronto.ca) by email if you have questions.

## Citation

`atlas` is an academic research software. If you use `atlas` in a scientific publication, please cite the following article.

```
@misc{atlas,

}
```
