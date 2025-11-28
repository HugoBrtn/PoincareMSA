# PoincaréMSA — quick start (updated)

This file is an updated README derived from the original `README.md` with a few practical additions and clarifications for running the code from the `examples/` folder.

Overview
--------
PoincaréMSA builds an interactive projection of an input protein multiple sequence alignment (MSA) on the Poincaré disk. It reproduces both local proximities and hierarchical structure in the data. The original project and algorithm are described in Susmelj et al. (see citations in the original README).

What changed in this README
---------------------------
- Clarified how to run the examples using precomputed embeddings (plain `.pt` files) or embeddings produced by an autoencoder.
- Documented how to start the pipeline from a precomputed distance matrix or an already computed RFA matrix.
- Updated environment setup instructions (the `env_poincare.yml` file has been refreshed). Includes a short note about installing JAX with the correct CUDA/JAX build.
- Added Git LFS usage notes (required for large example files in some setups).

Notebooks and examples
----------------------
The repository contains several notebooks in `examples/` and top-level Colab notebooks. The easiest way to try PoincaréMSA is to open one of the example notebooks (for instance `examples/kinases/PoincareMSA_kinases _method_choice.ipynb`) and follow the instructions there.

Three Colab versions are provided in the original README:
- `PoincareMSA_colab.ipynb` — general notebook (uses `.mfasta` by default)
- `PoincareMSA_colab_examples.ipynb` — runs example datasets shipped with the repo
- `PoincareMSA_colab_MMseqs2.ipynb` — builds MSAs using MMseqs2 and projects them

New: example input variants supported
------------------------------------
The example scripts/notebooks now support three alternative input workflows in addition to the original `.mfasta` pipeline:

1) Using precomputed embeddings (PLM-style `.pt` files)
   - In notebooks: set the data type to `plm` and point `in_name` / `embedding_path` to the directory containing `.pt` files.
   - From the command line: pass `--plm_embedding True` and `--input_path` pointing to a folder with `.pt` files.

2) Using PLM embeddings produced by an autoencoder (AAE)
   - In notebooks: set the data type to `plm_aae` and point to the proper embedding folder.
   - From the command line: use `--plm_embedding True` and the `--method` that corresponds to your workflow where appropriate (the notebooks set `data_type='plm_aae'`).

3) Starting from a precomputed distance matrix or an RFA matrix
   - You can start the pipeline with either a CSV distance matrix or a saved RFA matrix.
   - Notebooks: set `data_type` to `distance_matrix` or `RFA_matrix` and set the variables `distance_matrix`, `labels`, `mid_output` and `out_name_results` accordingly.
   - Command-line example (distance matrix):

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python scripts/build_poincare_map/main.py \
  --method distance_matrix \
  --distance_matrix path/to/distance_matrix.csv \
  --labels path/to/labels.csv \
  --matrices_output_path path/to/mid_output/ \
  --output_path path/to/results/ \
  --plm_embedding False \
  --knn 5 --sigma 1.0 --gamma 2.0 --seed 0
```

Command-line example (use precomputed RFA):

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python scripts/build_poincare_map/main.py \
  --method RFA_matrix \
  --matrices_output_path path/to/mid_output/ \
  --output_path path/to/results/ \
  --knn 5 --sigma 1.0 --gamma 2.0 --seed 0
```

Notes about labels and CSV formats
---------------------------------
- If you provide a distance matrix, also pass a `labels.csv` file with one identifier per line (or a CSV with a single column of labels). The code expects the number of labels to match the matrix dimension.
- The code has some robustness around CSV reading (it will try to detect when pandas misinterprets the first row as a header). If you get a mismatch error (labels length vs matrix size), check whether the distance matrix CSV has a header row or an extra index column. Re-save the CSV without header/index or provide a proper `labels.csv`.

Environment (conda) and JAX
---------------------------
The `env_poincare.yml` used by this repository has been updated. To update a local conda environment named `poincare` with the revised file run:

```bash
conda env update -n poincare -f env_poincare.yml --prune
```

JAX requires a matching `jaxlib` CUDA build for your local CUDA driver. The repository can't know your CUDA version, so install JAX with the wheel matching your CUDA version. An example (change `cuda11_cudnn82` to match your CUDA):

```bash
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# -> change cuda11_cudnn82 according to your CUDA version (cuda11_cudnn11, cuda12_cudnnX, ...)
```

If you don't have a GPU or don't want JAX GPU support, you can omit the `cuda` extras and install the CPU-only JAX:

```bash
pip install jax jaxlib
```

Git LFS (Large File Storage)
----------------------------
Some example datasets or embeddings may be stored with Git LFS. If you cloned the repository and see placeholder files or missing large blobs, install and pull LFS files:

1. Install git-lfs (Ubuntu/Debian example):

```bash
sudo apt update
sudo apt install git-lfs
git lfs install
```

2. In the project directory, fetch LFS-tracked blobs:

```bash
git lfs pull
```

Troubleshooting common issues
-----------------------------
- ValueError about "Number of sequence and number of annotation doesn't match": this happens when the annotation file has a different number of rows than the embedding CSV. Check `path_embedding` and `path_annotation` for correct files. The notebook includes a small diagnostic/fix helper to regenerate an annotation file aligned to the embeddings if needed.
- CSV header/shape issues when loading distance matrices: pandas may treat the first row as a header and drop a numeric row. If you see unexpected shapes, re-save the CSV without a header or pass `header=None` when generating it.

Development notes
-----------------
- The pipeline orchestration (I/O, option parsing) lives in `scripts/build_poincare_map/main.py`.
- RFA/KNN computation is implemented in `scripts/build_poincare_map/data.py`.
- Training code is implemented in `train.py`, `model.py` and `rsgd.py`.
- Visualization helpers are in `scripts/visualize_projection/`.

If you want me to (choose one):
- add an example invocation into each notebook that demonstrates starting from a distance matrix, or
- modify the example notebooks to automatically create a compatible `labels.csv` when missing.

References and contact
----------------------
Please refer to the original `README.md` for the publications and main contact emails.
