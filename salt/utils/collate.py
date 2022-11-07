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

    # labels is a dict of tensor so need to stack each
    labels_dict = {}
    elem = labels[0]
    for key in elem.keys():
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            # see https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L154 # noqa: E501
            numel = sum(x[key].numel() for x in labels)
            storage = elem[key].storage()._new_shared(numel, device=elem[key].device)
            out = (
                elem[key]
                .new(storage)
                .resize_(len(batch), *list(elem[key].size()))
                .long()  # assumes integer class label (TODO: generalise for regression)
            )
        labels_dict[key] = torch.stack([sample[key] for sample in labels], out=out)

    return torch.stack(inputs), torch.stack(valid), labels_dict
