uv run pyinstaller $1.py -y
cp .venv/lib/python3.12/site-packages/stable_baselines3/ -r dist/$1/_internal
cp .venv/lib/python3.12/site-packages/sb3_contrib/ -r dist/$1/_internal
cp .venv/lib/python3.12/site-packages/kaggle_environments/ -r dist/$1/_internal
