#!/usr/bin/env bash
tr -sc 'A-Za-z' '\n' < corpus.txt | tr A-Z a-z | sort | uniq -c | sort -n
