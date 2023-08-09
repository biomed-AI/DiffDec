# DiffDec: Structure-Aware Scaffold Decoration with an End-to-end Diffusion Model

## Summary
DiffDec is an end-to-end E(3)-equivariant diffusion model to optimize molecules through molecular scaffold decoration conditioned on the 3D protein pocket. 

This is a preliminary version of our code. We are currently in the process of cleaning up and organizing the code and will release it here as soon as possible.

<p align='center'>
<img src="./assets/overview.jpg" alt="architecture"/> 
</p>

## Install conda environment via conda yaml file
```bash
conda env create -f environment.yaml
```

## Datasets
Please refer to [`README.md`](./data/README.md) in the `data` folder.

## Sampling
You can sample 100 decorated compounds for each input scaffold and protein pocket and change the corresponding parameters in the script. Run the following:
```bash
bash sample.sh
```
You will get .xyz and .sdf files of the decorated compounds in the directory `sample_mols`. 

## Evaluation
You can run evaluation scripts after sampling decorated molecules:
```bash
bash evaluate.sh
```