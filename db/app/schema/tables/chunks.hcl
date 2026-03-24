table "chunks" {
  schema = schema.public

  column "id" {
    null = false
    type = integer
    identity {
      generated = ALWAYS
    }
  }

  column "document_id" {
    null = false
    type = integer
  }

  column "chunk_index" {
    null = false
    type = integer
  }

  column "content" {
    null = false
    type = text
  }

  column "token_count" {
    null = false
    type = integer
  }

  column "metadata" {
    null    = false
    type    = jsonb
    default = sql("'{}'::jsonb")
  }

  column "embedding" {
    null = false
    type = sql("vector(1536)")
  }

  column "created_at" {
    null    = false
    type    = timestamptz
    default = sql("now()")
  }

  primary_key {
    columns = [column.id]
  }

  foreign_key "chunks_document_id_fkey" {
    columns     = [column.document_id]
    ref_columns = [table.documents.column.id]
    on_delete   = CASCADE
  }

  unique "uq_chunks_document_id_chunk_index" {
    columns = [column.document_id, column.chunk_index]
  }

  index "ix_chunks_document_id" {
    columns = [column.document_id]
  }
}