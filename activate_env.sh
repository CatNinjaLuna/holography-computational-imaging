#!/bin/bash

# Activate the local virtual environment for holography project

echo "Activating holography virtual environment..."
source /Users/carolina1650/holography-computational-imaging/venv/bin/activate

echo "✓ Virtual environment activated!"
echo ""
echo "Installed packages:"
pip list | grep -E "(numpy|matplotlib|torch|odak|LightPipes)"
echo ""
echo "To deactivate, run: deactivate"
echo ""
echo "To run experiments:"
echo "  python -m phase1_physics_simulation.experiments.validation"
echo "  python -m phase1_physics_simulation.experiments.simulate_hologram"
echo "  python -m phase1_physics_simulation.experiments.backpropagation"
