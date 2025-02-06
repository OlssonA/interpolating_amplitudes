# Interpolating Amplitudes

This repository contains the implementations of the interpolation methods described in the paper [2412.09534](https://arxiv.org/abs/2412.09534), as well as the testfunctions used for the benchmarks.

**Authors**

V. Bresó-Pla, G. Heinrich, A. Olsson, V. Magerya

**Interpolation methods**  
The repository contains implementations of the following interpolation methods
* **Sparse Grids** (with modified extrapolating basis as well as extended not-a-knot B-spline and boundaryless B-spline basis functions)
* **B-splines** (uniform or not-a-knot constructions)
* **Polynomial Interpolation** (with Chebyshev polynomials)
* **Machine Learning** with MLP and Lorentz-Equivariant Geometric Algebra Transformers [(L-GATr)](https://arxiv.org/abs/2405.14806)

**Compilation of the testfunctions**  
To compile GoSam run..

## Machine Learning methods usage instructions

1. Create a virtual environment and install requirements
```bash
python -m venv venv
source venv/bin/activate
pip install -r ml/requirements.txt
```
2. Specify the home directory in the `base_dir` field inside the `config/config.yaml` file.
3. Run the command
```bash
"python run.py model={process}_{model}"
``` 
where the models are `gatr`, `mlp` and the processes are `qq_tth`, `qq_tth_loop`, `gg_tth`, `gg_tth_loop`, `gggh`.

**General comments**
* This implementation only runs with numpy data files where each row is structured as `[momenta, tree_level_amplitude, amplitude]`, where `tree_level_amplitude=amplitude` for all processes except `qq_tth_loop` and `gg_tth_loop`. We include examples sets with 100 points in the `data/` folder.
* This implementation is ready to run on CPU and GPU. It is advisable to run L-GATr networks on a GPU, otherwise processing large datasets can take several hours.
