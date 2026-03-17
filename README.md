## Setup

Create a virtual environment and install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Build the Cython extension:

    python setup.py build_ext --inplace

Run Task 3:

    mpiexec -n 8 python run_task3x.py --case all

Plot results:

    python plot_task3x.py --case all
