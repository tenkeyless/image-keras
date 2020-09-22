#!/bin/bash
usage="$(basename "$0") [-h | --help] [-r | --real]

where:
    -h | --help    Show this help text.
    -r | --real    Deploy to real PyPI.
"
real=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        #-r|--real) deploy="$2"; shift ;;
	-h|--help) echo "$usage" ; exit ;;
        -r|--real) real=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ $real -eq 1 ]
then
	echo "Deploy to real PyPI"
	python -m twine upload dist/*
else
	echo "Deploy to test PyPI"
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
fi
