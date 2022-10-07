"""Module docstring."""

import numpy as np


def test_docstring(x: float) -> float:
    """_summary_

    Parameters
    ----------
    x : float
        _description_

    Returns
    -------
    float
        _description_
    """
    return np.array(x)


def main():
    """test."""

    print(test_docstring(2))


if __name__ == "__main__":
    main()
