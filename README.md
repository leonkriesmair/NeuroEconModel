# Wang's Neuroeconomic Decision-Making Model

Python implementation of the biophysically realistic spiking neural network model from:

> Rustichini, A., & Padoa-Schioppa, C. (2015). A neuro-computational model of economic decisions. *Journal of Neurophysiology*, 114(3), 1382-1398.

This replicates the original MATLAB implementation (W11EC_JNP2015.m) of Wang's attractor network adapted for economic choice behavior in the orbitofrontal cortex (OFC).

## Model Architecture

The model implements a biophysically realistic attractor network with leaky integrate-and-fire neurons, winner-take-all competition, and realistic synaptic dynamics (NMDA, AMPA, GABA).

### Neural Populations

| Population | Role |
|---|---|
| **OV1, OV2** | Offer value cells -- encode value of options A and B |
| **CJ1, CJ2** | Chosen juice cells -- encode which option was chosen |
| **CV** | Chosen value cells -- encode value of chosen option |
| **NS** | Non-selective cells |
| **Inhibitory** | Interneurons providing competition |

### Key Parameters (Table 1 of paper)

- Network: NE=1600 excitatory, NI=400 inhibitory, f=0.15 structured fraction, w+=1.75 self-excitation
- Time constants: NMDA 100 ms, AMPA 2 ms, GABA 5 ms
- Simulation: dt=0.5 ms, trial duration 3000 ms, offer onset at 1000 ms

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: numpy, scipy, matplotlib.

## Quick Start

```bash
python quickstart.py
```

This runs a single trial and a short 50-trial simulation with behavioral output and a plot of neural activity.

### Single Trial

```python
from wang_model import WangModel
import numpy as np

model = WangModel()
result = model.run_trial(
    offA=10, offB=8,
    rangeA=(0, 15), rangeB=(0, 15),
    dj_HL=np.array([1.0, 1.0])
)

choice = "A" if result['nu1_wind'][-1] > result['nu2_wind'][-1] else "B"
print(f"Model chose: {choice}")
```

### Full Simulation

```python
from main_simulation import run_simulation

session = run_simulation(
    filesuffix='my_sim',
    session_mode='explicit',
    ntrials=500
)

print(f"Relative value: {session['behavData']['relvalue']:.3f}")
```

### Visualization

```python
from plotting import plot_session_summary
import pickle

with open('sessionSumm_my_sim.pkl', 'rb') as f:
    session = pickle.load(f)

plot_session_summary(session, save_dir='plots')
```

## File Structure

```
wang_model.py        Core model (WangModel class, LIF neurons, synaptic dynamics)
main_simulation.py   Run complete simulations
session_params.py    Generate offer sequences (explicit or parametric)
tuning_analysis.py   Compute tuning curves and behavioral metrics
plotting.py          Visualization (psychometric curves, trajectories, tuning maps)
quickstart.py        Minimal demo script
examples.py          Comprehensive usage examples
test_model.py        Automated validation tests
requirements.txt     Python dependencies
```

## Output Structure

Simulations save a `sessionSumm_*.pkl` file containing:

```python
{
    'params': {...},              # Simulation parameters
    'behavData': {
        'offerList': ...,         # Trial offers & choices
        'relvalue': ...,          # Relative subjective value
        'table01': ...            # Choice statistics
    },
    'tuning': {
        'rOV1': {...},            # OV1 tuning curves
        'r1': {...},              # CJ1 tuning curves
        'rI': {...},              # CV tuning curves
    },
    'traj_quant': {...},          # Quantile-averaged trajectories
    'CJ_traj_easysplit': {...},   # Easy vs split trial trajectories
    'CV_traj_chX': {...}          # CV trajectories by chosen option
}
```

## Performance

- Single trial: ~1-2 seconds
- 100 trials: ~2-3 minutes
- 500 trials: ~10-15 minutes

## Differences from MATLAB

- Random number generation (NumPy vs MATLAB RNG) produces different noise realizations
- Minor floating-point precision differences
- Python 0-based indexing vs MATLAB 1-based indexing
- Dictionaries replace MATLAB structs

## Citation

```bibtex
@article{rustichini2015neurocomputational,
  title={A neuro-computational model of economic decisions},
  author={Rustichini, Aldo and Padoa-Schioppa, Camillo},
  journal={Journal of Neurophysiology},
  volume={114},
  number={3},
  pages={1382--1398},
  year={2015}
}
```

## License

This implementation is provided for research and educational purposes.
