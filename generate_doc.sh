sphinx-apidoc -o doc/source image_keras/ --separate --ext-coverage -f
cd doc
make html
cd ..
doc2dash -n image_keras -f -i doc/dash_icon.png -A doc/build/html/
