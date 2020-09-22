sphinx-apidoc -o doc/source image-keras/ --separate --ext-coverage -f
cd doc
make html
cd ..
doc2dash -n image-keras -f -i doc/dash_icon.png -A doc/build/html/
