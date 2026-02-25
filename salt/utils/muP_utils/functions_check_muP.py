"""Adapted from the muP GitHub Repo
[1] https://github.com/microsoft/mup
Helper functions for performing coord check.
"""

from copy import copy
from pathlib import Path
from time import time

import pandas as pd
import torch

FDICT = {
    "l1": lambda x: torch.abs(x).mean(),
    "l2": lambda x: (x**2).mean() ** 0.5,
    "mean": lambda x: x.mean(),
    "std": lambda x: x.std(),
}


def convert_fdict(d):
    """Convert a dict `d` with string values to function values.

    Input:
        d: a dict whose values are either strings or functions
    Output:
        a new dict, with the same keys as `d`, but the string values are
        converted to functions using `FDICT`.
    Source: [1]
    """
    return dict([((k, FDICT[v]) if isinstance(v, str) else (k, v)) for k, v in d.items()])


def fix_batch_dataloader(batch, nsteps):
    """Create a fake dataloader with nsteps times the same batch.
    These nsteps versions are indepedent of each others for the
    inputs.

    This is required as our batch are list of dictionaries, so
    we need to deepcopy the batch to avoid having a pass of the model
    changing all batches in the fake dataloader.

    Note: copy.deecopy is not enough!
    """
    dataloader = []
    for _ in range(nsteps):
        new_batch = []
        for count, entry in enumerate(batch):
            new_entry = {}
            for key, val in entry.items():
                if count != 0:
                    new_entry[key] = val
                else:
                    new_entry[key] = val.clone().detach()
            new_batch.append(new_entry)
        dataloader.append(new_batch)
    return dataloader


def _record_coords(
    records, width, modulename, t, output_fdict=None, input_fdict=None, param_fdict=None
):
    """Returns a forward hook that records coordinate statistics.
    Adapted from muP GitHub Repo [1].

    Returns a forward hook that records statistics regarding the output, input,
    and/or parameters of a `nn.Module`. This hook is intended to run only once,
    on the timestep specified by `t`.

    On forward pass, the returned hook calculates statistics specified in
    `output_fdict`, `input_fdict`, and `param_fdict`, such as the normalized l1
    norm, of output, input, and/or parameters of the module. The statistics are
    recorded along with the `width`, `modulename`, and `t` (the time step) as a
    dict and inserted into `records` (which should be a list). More precisely,
    for each output, input, and/or parameter, the inserted dict is of the form

        {
            'width': width, 'module': modified_modulename, 't': t,
            # keys are keys in fdict
            'l1': 0.241, 'l2': 0.420, 'mean': 0.0, ...
        }

    where `modified_modulename` is a string that combines the `modulename` with
    an indicator of which output, input, or parameter tensor is the statistics
    computed over.

    The `*_fdict` inputs should be dictionaries with string keys and whose
    values can either be functions or strings. The string values are converted
    to functions via `convert_fdict`. The default values of `*_dict` inputs are
    converted to `output_fdict = dict(l1=FDICT['l1'])`, `input_fdict = {}`,
    `param_fdict = {}`, i.e., only the average coordinate size (`l1`) of the
    output activations are recorded.

    Inputs:
        records:
            list to append coordinate data to
        width:
            width of the model. This is used only for plotting coord check later
            on, so it can be any notion of width.
        modulename:
            string name of the module. This is used only for plotting coord check.
        t:
            timestep of training. This is used only for plotting coord check.
        output_fdict, input_fdict, param_fdict:
            dicts with string keys and whose values can either be functions or
            strings. The string values are converted to functions via
            `convert_fdict`
    Output:
        a forward hook that records statistics regarding the output, input,
        and/or parameters of a `nn.Module`, as discussed above.
    """
    output_fdict = {"l1": FDICT["l1"]} if output_fdict is None else convert_fdict(output_fdict)
    input_fdict = {} if input_fdict is None else convert_fdict(input_fdict)
    param_fdict = {} if param_fdict is None else convert_fdict(param_fdict)

    def f(module, myinput, myoutput):
        def get_stat(d, x, fdict):
            if isinstance(x, tuple | list):
                for i, _x in enumerate(x):
                    _d = copy(d)
                    _d["module"] += f"[{i}]"
                    get_stat(_d, _x, fdict)
            elif isinstance(x, dict):
                for name, _x in x.items():
                    _d = copy(d)
                    _d["module"] += f"[{name}]"
                    get_stat(_d, _x, fdict)
            elif isinstance(x, torch.Tensor):
                _d = copy(d)
                for fname, f in fdict.items():
                    _d[fname] = f(x).item()
                records.append(_d)
            else:
                raise NotImplementedError(f"Unexpected output type: {type(x)}")

        with torch.no_grad():
            ret = {"width": width, "module": modulename, "t": t}

            # output stats
            if isinstance(myoutput, tuple | list):
                for i, out in enumerate(myoutput):
                    _ret = copy(ret)
                    _ret["module"] += f":out[{i}]"
                    get_stat(_ret, out, output_fdict)
            elif isinstance(myoutput, dict):
                for name, out in myoutput.items():
                    _ret = copy(ret)
                    _ret["module"] += f":out[{name}]"
                    get_stat(_ret, out, output_fdict)
            elif isinstance(myoutput, torch.Tensor):
                _ret = copy(ret)
                for fname, f in output_fdict.items():
                    _ret[fname] = f(myoutput).item()
                records.append(_ret)
            else:
                raise NotImplementedError(f"Unexpected output type: {type(myoutput)}")

            # input stats
            if input_fdict:
                if isinstance(myinput, tuple | list):
                    for i, out in enumerate(myinput):
                        _ret = copy(ret)
                        _ret["module"] += f":in[{i}]"
                        get_stat(_ret, out, input_fdict)
                elif isinstance(myinput, dict):
                    for name, out in myinput.items():
                        _ret = copy(ret)
                        _ret["module"] += f":in[{name}]"
                        get_stat(_ret, out, input_fdict)
                elif isinstance(myinput, torch.Tensor):
                    _ret = copy(ret)
                    for fname, f in input_fdict.items():
                        _ret[fname] = f(myinput).item()
                    records.append(_ret)
                else:
                    raise NotImplementedError(f"Unexpected output type: {type(myinput)}")

            # param stats
            if param_fdict:
                for name, p in module.named_parameters():
                    _ret = copy(ret)
                    _ret["module"] += f":param[{name}]"
                    for fname, f in param_fdict.items():
                        _ret[fname] = f(p).item()
                    records.append(_ret)

    return f


def _get_coord_data(
    mup,
    models,
    dataloader,
    optcls,
    nsteps=3,
    filter_module_by_name=None,
    fix_data=True,
    cuda=False,
    nseeds=1,
    output_fdict=None,
    input_fdict=None,
    param_fdict=None,
    show_progress=True,
):
    """Inner method for `get_coord_data`, adapted from [1].

    Train the models in `models` with optimizer given by `optcls` and data from
    `dataloader` for `nsteps` steps, and record coordinate statistics specified
    by `output_fdict`, `input_fdict`, `param_fdict`. By default, only `l1` is
    computed for output activations of each module.

    Inputs:
        mup: bool,
            whether mup is to be used. If so, set_base_shape with rescale_params=rescale
        models:
            a dict of model paths, where the keys are numbers indicating width.
            Each entry of `models` is a path to an already instantiated model.
        dataloader:
            a dataloader, compatible with the global object classification task.
        optcls:
            a function so that `optcls(model)` gives an optimizer used to train
            the model.
        nsteps:
            number of steps to train the model
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        fix_data:
            Whether to fix the dataloader to a single batch. Default: True
        cuda:
            whether to use cuda or not. Default: False
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict:
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm.
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).
    """
    df = []
    if fix_data:
        saved_batch = next(iter(dataloader))

    if show_progress:
        from tqdm import tqdm

        pbar = tqdm(total=nseeds * len(models))

    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model_path in models.items():
            if fix_data:
                dataloader = fix_batch_dataloader(saved_batch, nsteps)
            model = torch.load(model_path, weights_only=False)
            model.train()
            if mup:
                from salt.utils.muP_utils.configuration_muP import instantiate_mup

                instantiate_mup(model, check=True)

            if cuda:
                model.cuda()
            optimizer = optcls(model.model)
            for batch_idx, batch in enumerate(dataloader, 1):
                remove_hooks = []
                # add hooks
                for name, module in model.named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(
                        module.register_forward_hook(
                            _record_coords(
                                df,
                                width,
                                name,
                                batch_idx,
                                output_fdict=output_fdict,
                                input_fdict=input_fdict,
                                param_fdict=param_fdict,
                            )
                        )
                    )

                if cuda:
                    batch.cuda()

                _, _, _, loss = model.shared_step(batch)
                optimizer.zero_grad()
                loss["loss"].backward()
                optimizer.step()

                # remove hooks
                for handle in remove_hooks:
                    handle.remove()

                if batch_idx == nsteps:
                    break
            if show_progress:
                pbar.update(1)
    if show_progress:
        pbar.close()
    return pd.DataFrame(df)


def get_coord_data(models, dataloader, optimizer="adamw", lr=None, mup=True, **kwargs):
    """Get coord data for coord check.

    Train the models in `models` with data from `dataloader` and optimizer
    specified by `optimizer` and `lr` for `nsteps` steps, and record coordinate
    statistics specified by `output_fdict`, `input_fdict`, `param_fdict`. By
    default, only `l1` is computed for output activations of each module.

    This function wraps around `_get_coord_data`, with the main difference being
    user can specify common optimizers via a more convenient interface.

    Inputs:
        models:
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a path to an already instantiated model.
        dataloader:
            a dataloader, compatible with the global object classification task.
        optimizer:
            a string in `['sgd', 'adam', 'adamw']`, with default being `'adamw'`.
        lr:
            learning rate. By default is 0.1 for `'sgd'` and 1e-2 for others.
        mup:
            If True, then use the optimizer from `mup.optim`; otherwise, use the
            one from `torch.optim`.
        filter_trainable_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be trained.
        nsteps:
            number of steps to train the model
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        fix_data:
            Whether to fix the dataloader to a single batch. Default: True
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict:
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm.
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).
    """
    if lr is None:
        lr = 0.1 if optimizer == "sgd" else 1e-2
    if mup:
        from mup.optim import MuAdam as Adam
        from mup.optim import MuAdamW as AdamW
        from mup.optim import MuSGD as SGD  # noqa: N814
    else:
        from torch.optim import SGD, Adam, AdamW

    def get_trainable(model):
        params = []
        for _, p in model.named_parameters():
            params.append(p)
        return params

    if optimizer == "sgd":

        def optcls(model):
            return SGD(get_trainable(model), lr=lr)

    elif optimizer == "adam":

        def optcls(model):
            return Adam(get_trainable(model), lr=lr)

    elif optimizer == "adamw":

        def optcls(model):
            return AdamW(get_trainable(model), lr=lr, weight_decay=1e-5)

    data = _get_coord_data(mup, models, dataloader, optcls, **kwargs)
    data["optimizer"] = optimizer
    data["lr"] = lr
    return data


def plot_coord_data(
    df,
    y="l1",
    save_to=None,
    suptitle=None,
    x="width",
    hue="module",
    legend="full",
    name_contains=None,
    name_not_contains=None,
    loglog=True,
    logbase=2,
    face_color=None,
):
    """Plot coord check data `df` obtained from `get_coord_data`.

    Input:
        df:
            a pandas DataFrame obtained from `get_coord_data`
        y:
            the column of `df` to plot on the y-axis. Default: `'l1'`
        save_to:
            path to save the resulting figure, or None. Default: None.
        suptitle:
            The title of the entire figure.
        x:
            the column of `df` to plot on the x-axis. Default: `'width'`
        hue:
            the column of `df` to represent as color. Default: `'module'`
        legend:
            'auto', 'brief', 'full', or False. This is passed to `seaborn.lineplot`.
        name_contains:
            only plot modules whose name contains `name_contains`
        name_not_contains:
            only plot modules whose name does not contain `name_not_contains`
        loglog:
            whether to use loglog scale. Default: True
        logbase:
            the log base, if using loglog scale. Default: 2
        face_color:
            background color of the plot. Default: None (which means white)
    Output:
        the `matplotlib` figure object
    """
    # preprocessing
    df = copy(df)
    # nn.Sequential has name '', which duplicates the output layer
    df = df[df.module != ""]
    try:
        if name_contains is not None:
            if not isinstance(name_contains, list):
                name_contains = [name_contains]
            df = df[df["module"].str.contains("|".join(name_contains))]
        if name_not_contains is not None:
            if not isinstance(name_not_contains, list):
                name_not_contains = [name_not_contains]
            df = df[~(df["module"].str.contains("|".join(name_not_contains)))]
        # for nn.Sequential, module names are numerical
        df["module"] = pd.to_numeric(df["module"])
    except Exception:  # noqa: S110, BLE001
        pass

    ts = df.t.unique()

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    def tight_layout(plt):
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plot
    fig = plt.figure(figsize=(5 * len(ts), 4))
    if face_color is not None:
        fig.patch.set_facecolor(face_color)
    for t in ts:
        plt.subplot(1, len(ts), t)
        sns.lineplot(x=x, y=y, data=df[df.t == t], hue=hue, legend=legend if t == 1 else None)
        plt.title(f"t={t}")
        if t != 1:
            plt.ylabel("")
        if loglog:
            plt.loglog(base=logbase)
    if suptitle:
        suptitle = (
            f"{suptitle} for layers {' & '.join(name_contains)}"
            if name_contains is not None
            else suptitle
        )
        suptitle = (
            f"{suptitle} without layers {' & '.join(name_not_contains)}"
            if name_not_contains is not None
            else suptitle
        )
        plt.suptitle(suptitle)
    tight_layout(plt)
    if save_to is not None:
        plt.savefig(save_to)
        print(f"Coordinate check plot saved to {save_to}")

    return fig


def _get_training_data(
    mup,
    models,
    dataloader,
    optcls,
    save_model=None,
    nsteps=10,
    cuda=True,
    nseeds=1,
    show_progress=True,
    noTQDM=False,
):
    """Inner method for `get_training_data`.

    Train the models in `models` with optimizer given by `optcls` and data from
    `dataloader` for `nsteps` steps, and record loss.

    Inputs:
        mup: bool,
            Whether mup is to be used. If so, set_base_shape with rescale_params=rescale
        models:
            a dict of model paths, where the keys are numbers indicating width.
            Each entry of `models` is a path to an already instantiated model.
        dataloader:
            a dataloader, compatible with the global object classification task.
        optcls:
            a function so that `optcls(model)` gives an optimizer used to train
            the model.
        save_model:
            path to save the model (if not None)
        nsteps:
            number of steps to train the model
        cuda:
            whether to try and use cuda. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        show_progress:
            show progress (either printed or TQDM, depending on next param)
        noTQDM:
            do not show progress using tqdm. Default: True
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'step', 'loss'`.
    """
    df = pd.DataFrame(columns=["width", "step", "loss"])

    def move_tensors_to_gpu(data):
        if isinstance(data, dict):
            return {key: move_tensors_to_gpu(value) for key, value in data.items()}
        if isinstance(data, list):
            return [move_tensors_to_gpu(item) for item in data]
        if isinstance(data, torch.Tensor):
            return data.cuda()
        return data

    if torch.cuda.is_available() and cuda:
        print("Using GPU")
        cuda = True
    else:
        print("Using CPU")
        cuda = False

    if save_model is not None:
        save_path = Path(save_model)
        save_path.mkdir(exist_ok=True)
        freq = 1 if nsteps <= 5 else round(nsteps / 5) if nsteps <= 1000 else round(nsteps / 10)

    freq_bar = 1 if nsteps <= 5 else round(nsteps / 10) if nsteps <= 1000 else round(nsteps / 100)
    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model_path in models.items():
            if show_progress and not noTQDM:
                from tqdm import tqdm

                pbar = tqdm(total=nsteps)

            model = torch.load(model_path, weights_only=False)
            model.train()

            if mup:
                from salt.utils.muP_utils.configuration_muP import instantiate_mup

                instantiate_mup(model, check=True)

            if cuda:
                model.cuda()
            optimizer = optcls(model.model)

            for batch_idx, batch in enumerate(dataloader, 1):
                t0 = time()
                if batch_idx < 10:
                    tf0 = time()

                mybatch = move_tensors_to_gpu(batch) if cuda else batch
                _, _, _, loss = model.shared_step(mybatch)
                optimizer.zero_grad()
                loss["loss"].backward()
                optimizer.step()

                myloss = loss["loss"].detach().cpu().numpy()
                df.loc[len(df)] = [width, batch_idx, myloss]

                if (save_model is not None) and (batch_idx % freq == 0):
                    torch.save(model, save_path / f"width_{width}_seed_{i}_step_{batch_idx}.pt")

                if show_progress and (batch_idx % freq_bar == 0):
                    t1 = time()
                    if not noTQDM:
                        pbar.update(freq_bar)
                    else:
                        print(
                            f"Step {batch_idx} / {nsteps} - loss = {myloss:.5} in time {t1 - t0:.3}"
                        )
                    t0 = t1
                if batch_idx < 10:
                    tf1 = time()
                    print(f"Iteration number {batch_idx} done in {tf1 - tf0:.3}")

                if batch_idx == nsteps:
                    break

            if save_model is not None:
                torch.save(model, save_path / f"width_{width}_seed_{i}_step_{batch_idx}.pt")
    if show_progress and not noTQDM:
        pbar.close()
    return df


def get_training_data(models, dataloader, optimizer="adamw", lr=None, mup=True, **kwargs):
    """Get training data.

    Train the models in `models` with data from `dataloader` and optimizer
    specified by `optimizer` and `lr` for `nsteps` steps, and record loss.

    This function wraps around `_get_coord_data`, with the main difference being
    user can specify common optimizers via a more convenient interface.

    Inputs:
        models:
            a dict of model paths, where the keys are numbers indicating width.
            Each entry of `models` is a path to an already instantiated model.
        dataloader:
            a dataloader, compatible with the jet classification task.
        optimizer:
            a string in `['sgd', 'adam', 'adamw']`, with default being `'adamw'`.
        lr:
            learning rate. By default is 0.1 for `'sgd'` and 1e-2 for others.
        mup:
            If True, then use the optimizer from `mup.optim`; otherwise, use the
            one from `torch.optim`.
        nsteps:
            number of steps to train the model
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        show_progress:
            show progress using tqdm.
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 'loss'`.
    """
    if lr is None:
        lr = 0.1 if optimizer == "sgd" else 1e-2
    if mup:
        from mup.optim import MuAdam as Adam
        from mup.optim import MuAdamW as AdamW
        from mup.optim import MuSGD as SGD  # noqa: N814
    else:
        from torch.optim import SGD, Adam, AdamW

    def get_trainable(model):
        params = []
        for _, p in model.named_parameters():
            params.append(p)
        return params

    if optimizer == "sgd":

        def optcls(model):
            return SGD(get_trainable(model), lr=lr)

    elif optimizer == "adam":

        def optcls(model):
            return Adam(get_trainable(model), lr=lr)

    elif optimizer == "adamw":

        def optcls(model):
            return AdamW(get_trainable(model), lr=lr, weight_decay=1e-5)

    return _get_training_data(mup, models, dataloader, optcls, **kwargs)


def plot_training_data(
    df,
    y="loss",
    save_to=None,
    suptitle=None,
    x="step",
    hue="width",
    legend="full",
    face_color=None,
    window_size=30,
):
    """Plot training check data `df` obtained from `get_training_data`.

    Input:
        df:
            a pandas DataFrame obtained from `get_coord_data`
        y:
            the column of `df` to plot on the y-axis. Default: `'l1'`
        save_to:
            path to save the resulting figure, or None. Default: None.
        suptitle:
            The title of the entire figure.
        x:
            the column of `df` to plot on the x-axis. Default: `'width'`
        hue:
            the column of `df` to represent as color. Default: `'width'`
        legend:
            'auto', 'brief', 'full', or False. This is passed to `seaborn.lineplot`.
        name_contains:
            only plot modules whose name contains `name_contains`
        name_not_contains:
            only plot modules whose name does not contain `name_not_contains`
        loglog:
            whether to use loglog scale. Default: True
        logbase:
            the log base, if using loglog scale. Default: 2
        face_color:
            background color of the plot. Default: None (which means white)
        window_size:
            an int indicating the size of the rolling average of the loss.
    Output:
        the `matplotlib` figure object
    """
    # preprocessing
    import numpy as np

    df = copy(df)

    # Combined the seeds
    df = df.groupby(["width", "step"], sort=False).agg({"loss": np.average})
    df = df.reset_index()

    # Now put a rolling average of loss, based on step for each width
    if window_size is not None:
        df = (
            df.groupby("width")
            .apply(
                lambda x: x.set_index("step")["loss"]
                .rolling(window=window_size, min_periods=1)
                .mean()
                .reset_index(name="loss")
            )
            .reset_index()
            .drop(columns="level_1")
        )

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    def tight_layout(plt):
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plot
    fig = plt.figure(figsize=(5, 4))
    if face_color is not None:
        fig.patch.set_facecolor(face_color)
    plt.subplot(1, 1, 1)
    sns.lineplot(x=x, y=y, data=df, hue=hue, legend=legend, palette="flare")
    if suptitle:
        plt.suptitle(suptitle)
    tight_layout(plt)
    if save_to is not None:
        plt.savefig(save_to)
        print(f"Coordinate check plot saved to {save_to}")

    return fig
