import os
import torch.multiprocessing as mp

AMA_PATH = "/data/kaltheuner/preprocessed-data/AMA"
DFAUST_PATH = "/data/kaltheuner/preprocessed-data/DFAUST"
DT4D_PATH = "/data/kaltheuner/preprocessed-data/DT4D"

OUT_BASE = "/data/kaltheuner/CVPR_2026"

DEFAULT_METHOD_CONFIG = "configs/method/fit_.json"


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # default datasets
    configs_runs = [
        "python Main.py --target obj -np 5000 -o {}/runs/ama/ -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/runs/dfaust/ -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + DFAUST_PATH,
        "python Main.py --target obj -np 5000 -o {}/runs/dt4d/ -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + DT4D_PATH,
    ]
    ###
    # Ablations
    ###
    # epochs
    from pathlib import Path

    EPOCHS_PATH = Path("configs/method/ablations/epochs")
    json_files = sorted(EPOCHS_PATH.glob("*.json"), key=lambda x: int(x.stem))
    configs_epochs = []
    for dataset in ["AMA", "DT4D", "DFAUST"]:
        for json_file in json_files:
            configs_epochs.append(
                "python Main.py --target obj -np 5000 -o {}/ablations/epochs/{}/{} -m {} --directory_path ".format(
                    OUT_BASE,
                    dataset,
                    json_file.stem,
                    str(json_file),
                )
                + (
                    AMA_PATH
                    if dataset == "AMA"
                    else DT4D_PATH if dataset == "DT4D" else DFAUST_PATH
                )
            )

    # input resolution
    configs_resolutions = [
        "python Main.py --target obj -np 1250 -o {}/ablations/resolution/1250 -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + AMA_PATH,
        "python Main.py --target obj -np 2500 -o {}/ablations/resolution/2500 -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + AMA_PATH,
        "python Main.py --target obj -np 10000 -o {}/ablations/resolution/10000 -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + AMA_PATH,
        "python Main.py --target obj -np 20000 -o {}/ablations/resolution/20000 -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + AMA_PATH,
    ]
    # sequence length
    configs_length = [
        # "python Main.py --target obj -np 5000 -dp /mnt/vci-gpu1-data/kaltheuner/processed_data/AMA/10/ -o {}/ablations/steps/10/ -m {}".format(
        #     OUT_BASE, DEFAULT_METHOD_CONFIG
        # ),
        # "python Main.py --target obj -np 5000 -dp /mnt/vci-gpu1-data/kaltheuner/processed_data/AMA/20/ -o {}/ablations/steps/20/ -m {}".format(
        #     OUT_BASE, DEFAULT_METHOD_CONFIG
        # ),
        "python Main.py --target obj -np 5000 -dp /mnt/vci-gpu1-data/kaltheuner/processed_data/AMA/40/ -o {}/ablations/steps/40/ -m {}".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        ),
        "python Main.py --target obj -np 5000 -dp /mnt/vci-gpu1-data/kaltheuner/processed_data/AMA/60/ -o {}/ablations/steps/60/ -m {}".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        ),
        "python Main.py --target obj -np 5000 -dp /mnt/vci-gpu1-data/kaltheuner/processed_data/AMA/80/ -o {}/ablations/steps/80/ -m {}".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        ),
        "python Main.py --target obj -np 5000 -dp /mnt/vci-gpu1-data/kaltheuner/processed_data/AMA/100/ -o {}/ablations/steps/100/ -m {}".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        ),
        "python Main.py --target obj -np 5000 -dp /mnt/vci-gpu1-data/kaltheuner/processed_data/AMA/120/ -o {}/ablations/steps/120/ -m {}".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        ),
    ]
    # time encoding
    configs_time = [
        "python Main.py --target obj -np 5000 -o {}/ablations/time_encoding/fourier1 -m configs/method/ablations/time_encodings/fourier_features1.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/time_encoding/fourier2 -m configs/method/ablations/time_encodings/fourier_features2.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/time_encoding/mlp -m configs/method/ablations/time_encodings/mlp.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/time_encoding/poly -m configs/method/ablations/time_encodings/poly.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/time_encoding/random_fourier -m configs/method/ablations/time_encodings/random_fourier.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/time_encoding/sinus_mlp -m configs/method/ablations/time_encodings/sinus_mlp.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/time_encoding/gaussian_fourier -m configs/method/ablations/time_encodings/gaussian_fourier.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
    ]
    # mlp design
    configs_mlp = [
        "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/1 -m configs/method/ablations/mlp_design/1.json  --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/2 -m configs/method/ablations/mlp_design/2.json  --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/3 -m configs/method/ablations/mlp_design/3.json  --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        # "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/4 -m configs/method/ablations/mlp_design/4.json  --directory_path ".format(
        #     OUT_BASE
        # )
        # + AMA_PATH,
        # "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/5 -m configs/method/ablations/mlp_design/5.json  --directory_path ".format(
        #     OUT_BASE
        # )
        # + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/6 -m configs/method/ablations/mlp_design/6.json  --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/7 -m configs/method/ablations/mlp_design/7.json  --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/8 -m configs/method/ablations/mlp_design/8.json  --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/9 -m configs/method/ablations/mlp_design/9.json  --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/10 -m configs/method/ablations/mlp_design/10.json  --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/mlp_design/11 -m configs/method/ablations/mlp_design/11.json  --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
    ]

    configs_rotations = [
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/75/cayley -m configs/method/ablations/rotations/75/cayley.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/75/quaternions -m configs/method/ablations/rotations/75/quaternions.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/75/exp -m configs/method/ablations/rotations/75/exp.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/125/cayley -m configs/method/ablations/rotations/125/cayley.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/125/quaternions -m configs/method/ablations/rotations/125/quaternions.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/125/exp -m configs/method/ablations/rotations/125/exp.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/250/cayley -m configs/method/ablations/rotations/250/cayley.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/250/quaternions -m configs/method/ablations/rotations/250/quaternions.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/250/exp -m configs/method/ablations/rotations/250/exp.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/500/cayley -m configs/method/ablations/rotations/500/cayley.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/500/quaternions -m configs/method/ablations/rotations/500/quaternions.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/500/exp -m configs/method/ablations/rotations/500/exp.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/1000/cayley -m configs/method/ablations/rotations/1000/cayley.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/1000/quaternions -m configs/method/ablations/rotations/1000/quaternions.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/rotations/1000/exp -m configs/method/ablations/rotations/1000/exp.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
    ]

    configs_instant_ngp = [
        "python Main.py --target obj -np 5000 -o {}/ablations/instant_ngp/ama/ --instant_ngp -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/instant_ngp/dfaust/ --instant_ngp -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + DFAUST_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/instant_ngp/dt4d/ --instant_ngp -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + DT4D_PATH,
    ]

    configs_stability_delta = [
        "python Main.py --target obj -np 5000 -o {}/ablations/stability/delta/pdg/ -m configs/method/ablations/stability_function/delta/pdg.json --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/stability/delta/exp/ -m configs/method/ablations/stability_function/delta/exp.json --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/stability/delta/lerp/ -m configs/method/ablations/stability_function/delta/lerp.json --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/stability/delta/linear/ -m configs/method/ablations/stability_function/delta/linear.json --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/stability/delta/none/ -m configs/method/ablations/stability_function/delta/none.json --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
    ]

    configs_stability_confidence = [
        "python Main.py --target obj -np 5000 -o {}/ablations/stability/conf/pdg/ -m configs/method/ablations/stability_function/conf/pdg.json --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/stability/conf/direct/ -m configs/method/ablations/stability_function/conf/direct.json --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/stability/conf/delta_based/ -m configs/method/ablations/stability_function/conf/delta_based.json --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
        "python Main.py --target obj -np 5000 -o {}/ablations/stability/conf/constant/ -m configs/method/ablations/stability_function/conf/constant.json --directory_path ".format(
            OUT_BASE
        )
        + "/data/kaltheuner/processed_data/AMA/40/",
    ]

    configs_method = [
        "python Main.py --target obj -np 5000 -o {}/ablations/method/single_resolution/  -m {} --directory_path ".format(
            OUT_BASE,
            "/data/kaltheuner/git/preconditioned-deformation-nets/configs/method/ablations/method/single_res.json",
        )
        + DT4D_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/method/wo_isomentry/  -m {} --directory_path ".format(
            OUT_BASE,
            "/data/kaltheuner/git/preconditioned-deformation-nets/configs/method/ablations/method/wo_iso.json",
        )
        + DT4D_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/method/wo_preconditioning/  -m {} --directory_path ".format(
            OUT_BASE,
            "/data/kaltheuner/git/preconditioned-deformation-nets/configs/method/ablations/method/wo_prec.json",
        )
        + DT4D_PATH,
        "python Main.py --target obj -np 5000 -o {}/ablations/method/instant_ngp/ --instant_ngp -m {} --directory_path ".format(
            OUT_BASE, DEFAULT_METHOD_CONFIG
        )
        + DT4D_PATH,
    ]

    #####
    # Supplemental
    #####
    supls_grid_lvl = [
        # "python Main.py --target obj -np 5000 -o {}/supplemental/grid_level/one -m configs/method/supplemental/Grid_Level/one.json --directory_path ".format(
        #     OUT_BASE
        # )
        # + AMA_PATH,
        # "python Main.py --target obj -np 5000 -o {}/supplemental/grid_level/two -m configs/method/supplemental/Grid_Level/two.json --directory_path ".format(
        #     OUT_BASE
        # )
        # + AMA_PATH,
        # "python Main.py --target obj -np 5000 -o {}/supplemental/grid_level/four -m configs/method/supplemental/Grid_Level/four.json --directory_path ".format(
        #     OUT_BASE
        # )
        # + AMA_PATH,
        # "python Main.py --target obj -np 5000 -o {}/supplemental/grid_level/six -m configs/method/supplemental/Grid_Level/six.json --directory_path ".format(
        #     OUT_BASE
        # )
        # + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/supplemental/grid_level/ten -m configs/method/supplemental/Grid_Level/ten.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/supplemental/grid_level/twelve -m configs/method/supplemental/Grid_Level/twelve.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
    ]

    supls_lambda_weight = [
        "python Main.py --target obj -np 5000 -o {}/supplemental/lambda_weight/zero -m configs/method/supplemental/lambda_weight/zero.json --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        # "python Main.py --target obj -np 5000 -o {}/supplemental/lambda_weight/low -m configs/method/supplemental/lambda_weight/low.json --directory_path ".format(
        #     OUT_BASE
        # )
        # + AMA_PATH,
        # "python Main.py --target obj -np 5000 -o {}/supplemental/lambda_weight/high -m configs/method/supplemental/lambda_weight/high.json --directory_path ".format(
        #     OUT_BASE
        # )
        # + AMA_PATH,
        # "python Main.py --target obj -np 5000 -o {}/supplemental/lambda_weight/very_high -m configs/method/supplemental/lambda_weight/very_high.json --directory_path ".format(
        #     OUT_BASE
        # )
        # + AMA_PATH,
    ]

    supls_noise = [
        "python Main.py --target obj -np 5000 -o {}/supplemental/noise/025 --noise 0.25 --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/supplemental/noise/05 --noise 0.5 --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/supplemental/noise/1 --noise 1  --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
        "python Main.py --target obj -np 5000 -o {}/supplemental/noise/2 --noise 2 --directory_path ".format(
            OUT_BASE
        )
        + AMA_PATH,
    ]
    # configs = configs_epochs + configs_grid_lvl + configs_resolutions + configs_length + configs_time + configs_mlp + configs_rotations +configs_stability_delta
    configs = supls_grid_lvl
    for c in configs:
        os.system(c + " --device 0 1 2 3 4 7 6")
