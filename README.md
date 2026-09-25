
```{code-cell}
# The interactive jupyer book is here: https://dorksquith.github.io/DataAnalysisTechniques

# Clone the repository
git clone https://github.com/dorksquith/DataAnalysisTechniques.git
cd DataAnalysisTechniques

# Pick up any recent changes since you cloned
git pull


# Create a virtual python environment - see https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
python3 -m venv .venv

# activate the virtual python environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# run the jupyter book and open in browser
jupyter book start --execute
```

