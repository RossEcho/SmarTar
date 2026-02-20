#!/data/data/com.termux/files/usr/bin/bash

DB=$1
ARCH=$2
QUERY=$3

python src/search.py "$DB" "$QUERY" --paths --limit 1 | \
xargs -I{} python src/extract.py "$ARCH" {} --out extracted
