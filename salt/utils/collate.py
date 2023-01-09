import torch


def collate(batch):
    """Custom batch collate function.

    Parameters
    ----------
    batch : list
        List of tuple of batch items

    Returns
    -------
    tuple
        Concated batch items
    """

    inputs = [sample[0] for sample in batch]
    valid = [sample[1] for sample in batch]
    labels = [sample[2] for sample in batch]

    inputs_dict = {}
    valid_dict = {}
    elem_inputs = inputs[0]
    elem_valid = valid[0]
    for key in elem_inputs.keys():
        out_i = None
        out_v = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            # see https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L154 # noqa: E501
            i_val = elem_inputs[key]
            n_inputs = sum(x[key].numel() for x in inputs)
            s_inputs = i_val.storage()._new_shared(n_inputs, device=i_val.device)
            out_i = i_val.new(s_inputs).resize_(len(batch), *list(i_val.size()))
            if key in elem_valid:
                v_val = elem_valid[key]
                n_valid = sum(x[key].numel() for x in valid)
                s_valid = v_val.storage()._new_shared(n_valid, device=v_val.device).float()
                out_v = i_val.new(s_valid).resize_(len(batch), *list(v_val.size())).bool()
        inputs_dict[key] = torch.stack([sample[key] for sample in inputs], out=out_i)
        if key in elem_valid:
            valid_dict[key] = torch.stack([sample[key] for sample in valid], out=out_v)

    labels_dict = {}
    elem = labels[0]
    for key in elem.keys():
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum(x[key].numel() for x in labels)
            storage = elem[key].storage()._new_shared(numel, device=elem[key].device)
            out = (
                elem[key]
                .new(storage)
                .resize_(len(batch), *list(elem[key].size()))
                .long()  # assumes integer class label (TODO: generalise for regression)
            )
        labels_dict[key] = torch.stack([sample[key] for sample in labels], out=out)

    return inputs_dict, valid_dict, labels_dict
