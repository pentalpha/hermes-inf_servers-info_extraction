
FROM="./"
TO="hermes-ner-b.southamerica-east1-a.poc-mj-474517:/home/pita/triton-info-extraction"

# Padrões a ignorar
EXCLUDES=(
  ".git"
  ".gitignore"
  ".venv"
  "venv"
  "__pycache__"
  "*.pyc"
  ".DS_Store"
  "node_modules"
  ".idea"
  ".vscode"
  "*.swp"
  "tmp"
  "dist"
  "build"
  ".mypy_cache"
  ".pytest_cache"
  "upload.sh"
  "sqlite.db/*"
  "sqlite.db"
  "results/*"
  "logs/*"
  "*.log"
)

RSYNC_OPTS=(-zz -zarv --progress)

# Construir argumentos --exclude
EXCLUDE_ARGS=()
for e in "${EXCLUDES[@]}"; do
  EXCLUDE_ARGS+=(--exclude="$e")
done

echo "Sincronizando:"
echo "  FROM: $FROM"
echo "  TO:   $TO"
echo

# Executar rsync
/usr/bin/rsync "${RSYNC_OPTS[@]}" "${EXCLUDE_ARGS[@]}" "$FROM" "$TO"