"""Top level training script, powered by the lightning CLI."""

import sys

from salt.main import main as main_salt
from salt.utils.muP_utils.configuration_muP import (
    generate_base_delta_config,
    store_shapes_mup,
)
from salt.utils.muP_utils.generateModel import generateModel


def main(args=None):
    """Main script for muP.
    - Should be used to instantiate needed base & delta models.
    - Can also execute the main training if global runFit is True.
    - Can also perform a check muP is working if global check is True.

    Note: recommended usage is to only use this script to instantiate
    the base & delta models. You can then run a normal salt for the main
    models, which is going to load the base & delta models.
    """
    runFit = False
    check = False
    if args is not None:
        sys.argv = [""]
        for item in args:
            if "=" in item:
                sys.argv += [item.split("=")[0]]
                sys.argv += [item.split("=")[1]]
            else:
                sys.argv += [item]

    if "--runFit" in sys.argv:
        sys.argv.remove("--runFit")
        runFit = True
    if "--check" in sys.argv:
        sys.argv.remove("--check")
        check = True
    train_file, norm_dict = None, None
    if "--data.train_file" in sys.argv:
        ind = sys.argv.index("--data.train_file")
        train_file = sys.argv[ind + 1]
    if "--data.norm_dict" in sys.argv:
        ind = sys.argv.index("--data.norm_dict")
        norm_dict = sys.argv[ind + 1]

    full_args = sys.argv[1:]
    config = sys.argv[2]
    restargs = [""]
    if len(sys.argv) > 2:
        restargs = sys.argv[3:]

    print("-" * 100)
    print("Initialising base and delta shapes")
    base_config, store_path = generate_base_delta_config(config, "base")
    delta_config, _ = generate_base_delta_config(config, "delta")

    # Execute generateModel to get base_model saved at a path in base_config
    generateModel(args=["base", "--trainer.callbacks", "", f"--config={base_config}", *restargs])
    generateModel(args=["delta", "--trainer.callbacks", "", f"--config={delta_config}", *restargs])

    # Load the models and save the shapes in filename
    store_shapes_mup(path=store_path)

    # If you want to check muP works
    if check:
        from salt.utils.muP_utils.configuration_muP import (
            check_muP,
            generate_config_muPtest,
            get_models_muPtest,
        )

        print("Instantiating muP check models")

        # Need also base and delta without wrapper for local training
        generateModel(
            args=["temp_base", "--trainer.callbacks", "", f"--config={base_config}", *restargs]
        )
        generateModel(
            args=["temp_delta", "--trainer.callbacks", "", f"--config={delta_config}", *restargs]
        )

        store_shapes_mup(path=store_path, check=check)

        # Create the variations
        variations = [32, 64, 128, 256, 512, 1024, 2056]
        configs = generate_config_muPtest(config, variations)
        configs.extend(generate_config_muPtest(config, variations, muP=False))
        for enum, mod_config in enumerate(configs):
            temp_add = "muP" if "muP" in str(mod_config.name) else "sP"

            generateModel(
                args=[
                    f"temp_{temp_add}_{enum % len(variations)}",
                    "--trainer.callbacks",
                    "",
                    f"--config={mod_config}",
                    *restargs,
                ]
            )

        if runFit:
            print("-" * 100)
            print("Running muP checks")
            assert train_file is not None, "You must give a valid train_file path"
            assert norm_dict is not None, "You must give a valid norm_dict path"

            # Retrieve the muP and SP models
            models_muP, models_sP = get_models_muPtest(variations)

            # Perform checks:
            #   - Warning, will train the models!
            #       Recommended to use a script to separately do the training
            #   - requires to manually pass the train_file and norm_dict.
            #       The "variables" dictionary is set in
            check_muP(
                models_muP,
                models_sP,
                num_train=2000,
                nsteps_training=1500,
                nsteps_coocheck=3,
                nseeds=5,
                batch_size=1000,
                doCoord=True,
                doTraining=True,
                train_file=train_file,
                norm_dict=norm_dict,
            )
        return

    # Normal execution of salt for the real model, called model.
    # The lightning_module should be able to collect filename and do:
    #       set_base_shapes(model, base_model, delta=delta_model)
    if runFit:
        print("-" * 100)
        print("Starting main training")
        main_salt(["fit", *full_args])

    # To cleam the muP produced files
    # clean_environment()


if __name__ == "__main__":
    main()
