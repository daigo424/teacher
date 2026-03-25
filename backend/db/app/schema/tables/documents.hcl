table "documents" {
  schema = schema.public

  column "id" {
    null = false
    type = integer
    identity {
      generated = ALWAYS
    }
  }

  column "title" {
    null = false
    type = varchar(255)
  }

  column "source_path" {
    null = false
    type = varchar(1024)
  }

  column "checksum" {
    null = false
    type = varchar(64)
  }

  column "created_at" {
    null    = false
    type    = timestamptz
    default = sql("now()")
  }

  column "updated_at" {
    null    = false
    type    = timestamptz
    default = sql("now()")
  }

  primary_key {
    columns = [column.id]
  }

  unique "documents_source_path_key" {
    columns = [column.source_path]
  }
}