rm -rf dist
python3 -m build
pip install dist/*.whl --force-reinstall