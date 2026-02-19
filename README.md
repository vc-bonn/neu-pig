<h1 align="center" id="heading">Preconditioned Deformation Grids</h1>

<p align="center">
    <p align="center">
		<b><a href="https://cg.cs.uni-bonn.de/person/m-sc-julian-kaltheuner">Julian Kaltheuner</a></b>
        &nbsp;·&nbsp;
		<b><a href="https://cg.cs.uni-bonn.de/person/dr-markus-plack">Markus Plack</a></b>
        &nbsp;·&nbsp;
		<b><a href="https://cg.cs.uni-bonn.de/person/dr-hannah-droege">Hannah Droege</a></b>
        &nbsp;·&nbsp;
		<b><a href="https://cg.cs.uni-bonn.de/person/dr-patrick-stotko">Patrick Stotko</a></b>
        &nbsp;·&nbsp;
        <b><a href="https://cg.cs.uni-bonn.de/person/prof-dr-reinhard-klein">Reinhard Klein</a></b>
    </p>
    <p align="center">
       University of Bonn &nbsp
    </p>
    <h3 align="center">CVPR 2026</h3>
    <h3 align="center">
        <a href="https://arxiv.org/abs/2509.18097">Paper (arxiv)</a>
		</h3>
    <div align="center"></div>
</p>

![](assets/teaser.png)

## Install
```
git clone https://github.com/vc-bonn/neu-pig.git neu-pig
cd neu-pig

git submodule update --init

conda create -n neupig python=3.12
conda activate neupig


conda install -y -c conda-forge openblas cmake ninja
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install ext/pcgrid
pip install tqdm tensorboard scikit-learn charonload cmake gpytoolbox imageio matplotlib ninja open3d opencv-python pykdtree trimesh pymeshlab openmesh tensorboardx gitpython rich
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Main.py Arguments
```
python Main.py --...
[-m / --methodConfig] - Path to the method config file (str)
[-se / --seed] - random seed (int)
[-d / --device] - cuda device number (int)
[-t / --target] - which target type to use. If pcl is chosen, make sure that the directory path is as configured in the AMA dataset. ("pcl"/"obj")
[-np / --number_points] - if [-t == "obj"] set the number of target points sampled per obj file (int)
[-o / --out_path] - output directory (str)
[-dp / --directory_path] - path to the input objects, see PATH-STRUCTURE section (str)
[-s / --skip] - how many objects to skip (int)
[-ns / --noise] - standard deviation of gaussian noise added to target points (float)
```

## PATH-STRUCTURE
To download the preprocessed data, see the data section of [Dynosurf](https://github.com/yaoyx689/DynoSurf?tab=readme-ov-file).
``` 
--AMA
|
|--crane_0010
|-|
|-|--gt
|-|--pcl_seqs
|-|--points_clouds
|
|--crane_0027
|-|
|-|--gt
|-|--pcl_seqs
|-|--points_clouds
```

## RUNS
All run configs are predefined in configs/method/runs. For the ablation study see the configs/method/ablations path. Alter the path arguments accordingly [-o / -dp]. To create the extended AMA dataset, without sequences in the range of 40-120 frames, see src/io/dataset/process_ama_dataset.py.

For the chained execution we supply a run.py script, see src/run.py. See the --help arguments for all the run/supplemental run options. We support multi-gpu runs, setting multiple gpu ids via --devices.

To run the initialization methods from DynoSurf or Motion2VecSets, make sure to install their required packages. For Motion2VecSets, we recommend creating a separate conda environment with the appropriate Python version to avoid version conflicts. For the execution of DynoSurf, please refer to their [repository](https://github.com/yaoyx689/DynoSurf), or execute src/related_work/run_dynosurf.py.
Paths need to be altered according to your setup.

