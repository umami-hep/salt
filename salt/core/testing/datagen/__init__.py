"""Schema-driven synthetic test-data generation for salt/core.

Public API
----------
* ``generate_data(schema, flags=None)`` -> ``{group: structured ndarray}``
* ``write_h5(data, path, attrs=None, schema=None)``
* ``compute_norm_dict(data, schema=None)``
* ``compute_class_dict(data, schema=None, flags=None)``
* ``load_schema(path)``
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
