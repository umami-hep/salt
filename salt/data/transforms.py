import numpy as np


class GaussianNoise:
    def __init__(self, noise_params=None):
        """Initializes the Gaussian noise generator with customizable parameters.

        Parameters
        ----------
        noise_params : dict, optional
            A dictionary where keys are input names (e.g., "tracks", "jets") and values are
            dictionaries with "mean", "std", and optionally a "mask" array for selective noise.
        """
        self.noise_params = noise_params if noise_params is not None else {}

    def add_noise(self, data, mean, std):
        """Add Gaussian noise to specified subfields of a dataset.

        Parameters
        ----------
        data : Numpy array
            Array containing data to which noise will be added
        mean : float
            Mean of noise to be added
        std: : float
            Standard deviation of noise to be added

        Returns
        -------
        data : Numpy array
            Data with noise added
        """
        # Generate noise
        rng = np.random.default_rng()
        noise = rng.normal(mean, std, data.shape)

        # Scale noise based on data and apply it
        data_noise = data * noise
        data += data_noise

        return data

    def __call__(self, data, input_type):
        """Applies Gaussian noise to all variables of the given type as specified in `noise_params`.

        Parameters
        ----------
        data : Numpy array
            Array containing data for input variables
        input_type : str
            Name of the input variable type e.g. jets or tracks

        Returns
        -------
        data : Numpy array
            The modified data array with noise applied
        """
        for noise_dict in self.noise_params:
            # Skip if wrong input type
            if noise_dict["input_type"] != input_type:
                continue
            var = noise_dict["variable"]
            mean = noise_dict["mean"]
            std = noise_dict["std"]
            noise_data = data[var]
            data[var] = self.add_noise(noise_data, mean, std)
        return data
