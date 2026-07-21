"""Schema-driven synthetic test-data generation for salt: ``generate_data``,
``write_h5``, ``compute_norm_dict``, ``compute_class_dict``, ``load_schema``.
"""

from .engine import generate_data
from .io import compute_class_dict, compute_norm_dict, write_h5
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

__all__ = [
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
    "write_h5",
]
