data "hcl_schema" "app" {
  paths = fileset("schema/**/*.hcl")
}

env "local" {
  url = getenv("APP_DB_URL")
  dev = getenv("APP_DEV_DB_URL")
  src = data.hcl_schema.app.url
}