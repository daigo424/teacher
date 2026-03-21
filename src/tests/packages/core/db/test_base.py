from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from packages.core.db.base import Base


def test_base_type_annotation_map_uses_postgres_types():
    assert Base.type_annotation_map[dict] is JSONB
    assert isinstance(Base.type_annotation_map[list[float]], Vector)
