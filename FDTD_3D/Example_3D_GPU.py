"""Million-cell, device-resident GPU version of the 3D scattering example.

This runs the same material geometry, plane excitation, HDF5 monitor, power
spectrum, and 3D dB NF2FF calculation as ``Example_3D.py``.  Only the backend
and default output directory differ, making CPU/GPU comparisons straightforward.
"""

try:
    from .Example_3D import run_example
except ImportError:
    from Example_3D import run_example


if __name__ == "__main__":
    run_example("gpu")
