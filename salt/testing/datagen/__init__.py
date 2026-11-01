"""Schema-driven synthetic test-data generation for salt: ``write_dataset`` (one call),
``generate_data``, ``write_h5``, ``compute_norm_dict``, ``compute_class_dict``, ``load_schema``.
"""

from pathlib import Path

from .engine import generate_data
from .io import compute_class_dict, compute_norm_dict, write_dataset, write_h5
from .pipeline import Pipeline, RecipeError, load_pipeline
from .schema import (
    DistributionField,
    GroupSpec,
    IdField,
    LabelField,
    LinkField,
    Schema,
    SchemaError,
    as_schema,
    load_schema,
    parse_schema,
)

#: The synthetic salt-v1 training-file schema (jets/tracks/flows/electrons/truth_hadrons
#: + aliases; flags ``make_xbb``/``inc_taus``/``inc_params``/``is_gn3``).
SALT_V1_SCHEMA: Path = Path(__file__).parent / "recipes" / "salt-v1-schema.yaml"

__all__ = [
    "SALT_V1_SCHEMA",
    "DistributionField",
    "GroupSpec",
    "IdField",
    "LabelField",
    "LinkField",
    "Pipeline",
    "RecipeError",
    "Schema",
    "SchemaError",
    "as_schema",
    "compute_class_dict",
    "compute_norm_dict",
    "generate_data",
    "load_pipeline",
    "load_schema",
    "parse_schema",
    "write_dataset",
    "write_h5",
]
