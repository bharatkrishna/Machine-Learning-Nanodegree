#!/usr/bin/env bash

# brew install pandoc-crossref

pandoc -F pandoc-crossref -V geometry:margin=1in --variable papersize=letter -s project_report.md -o project_report.pdf
