"""To rapidly train one of the model saved for mup."""

import argparse

from salt.utils.muP_utils.configuration_muP import check_mup, get_models_muptest

TRAIN = "./pp_output_train.h5"
NORM = "./norm_dict.yaml"
SAVE = "./plots_mup"


def mup_one_model_run(
    modInd, modType, variations, lr, nsteps, nseeds, num_workers, batch_size, num_train
):
    models_mup, models_sP = get_models_muptest(variations, modInd, modType)
    save = f"{SAVE}_lr_{lr}_batch_{batch_size}"
    check_mup(
        models_mup,
        models_sP,
        lr=lr,
        nsteps=nsteps,
        nseeds=nseeds,
        num_train=num_train,
        batch_size=batch_size,
        num_workers=num_workers,
        train_file=TRAIN,
        norm_dict=NORM,
        save_path=save,
        doPlots=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional app description")
    parser.add_argument("--modInd", type=int, required=True, help="Ind for the model index")
    parser.add_argument("--modType", type=int, required=True, help="1 for mup, 0 for sP")
    parser.add_argument(
        "--variations", type=int, nargs="+", required=True, help="1 for mup, 0 for sP"
    )
    parser.add_argument("--lr", type=float, default=1e-4, required=False)
    parser.add_argument("--nsteps", type=int, default=2, required=False)
    parser.add_argument("--nseeds", type=int, default=1, required=False)
    parser.add_argument("--num_workers", type=int, default=0, required=False)
    parser.add_argument("--batch_size", type=int, default=500, required=False)
    parser.add_argument("--num_train", type=int, default=-1, required=False)
    args = parser.parse_args()
    mup_one_model_run(
        args.modInd,
        args.modType,
        args.variations,
        args.lr,
        args.nsteps,
        args.nseeds,
        args.num_workers,
        args.batch_size,
        args.num_train,
    )
