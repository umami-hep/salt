from pathlib import Path

import pytest

from salt.to_onnx import main as to_onnx
from salt.utils.get_onnx_metadata import main as get_onnx_metadata

w1 = "ignore::UserWarning:"
w2 = "ignore::torch.jit.TracerWarning:"


@pytest.mark.filterwarnings(w1)
@pytest.mark.filterwarnings(w2)
class TestONNX:
    test_dir = Path(__file__).resolve().parent
    config_path = Path(test_dir.parent / "configs")
    tmp_dir = Path("/tmp/salt_tests/")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = None
    sd_path = None
    ckpt_path = None
    onnx_path = None

    @classmethod
    def setup_class(cls):
        """Setup tests (runs once).

        look in the test dir for a training run.
        """
        dirs = [x for x in cls.tmp_dir.iterdir() if x.is_dir()]
        test_dir = [x for x in dirs if (x / "config.yaml").exists()][0]
        config_path = test_dir / "config.yaml"
        sd_path = [x for x in test_dir.iterdir() if x.suffix == ".json"][0]
        ckpt_path = list((test_dir / "ckpts").iterdir())[-1]

        cls.test_dir = test_dir
        cls.config_path = config_path
        cls.ckpt_path = ckpt_path
        cls.sd_path = sd_path
        cls.onnx_path = test_dir / "model.onnx"

    @pytest.mark.depends(name="to_onnx", on=["train"])
    def test_to_onnx(self) -> None:
        args = []
        args += [f"--config={self.config_path}"]
        args += [f"--ckpt_path={self.ckpt_path}"]
        args += ["--track_selection=dipsLoose202102"]
        args += ["--overwrite"]
        args += [f"--sd_path={self.sd_path}"]
        to_onnx(args)

    @pytest.mark.depends(on=["to_onnx"])
    def test_get_onnx_metadata(self) -> None:
        args = [str(self.onnx_path)]
        get_onnx_metadata(args)
