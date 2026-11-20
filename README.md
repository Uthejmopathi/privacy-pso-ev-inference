# A2P2 - A Privacy-Constrained Adaptive Inference Framework for On-Device Multimodal AI Under Dynamic Energy and Thermal Budgets

Author: Uthej (author 2, paper 2), Track 2 - in-vehicle AI assistant in EVs under battery and thermal budgets.
Method: Particle Swarm Optimization of an adaptive execution-mode policy (full / int8 / secure offload / defer) with a hard
privacy mask and energy/thermal penalties, using LightGBM outcome surrogates. Baselines: always-local, always-offload,
threshold policy, random search (plus the logged field controller).

Contents
- data/    synthetic dataset A2P2_ev_assistant_inference.csv + data dictionary (README)
- code/    run_all.py (full pipeline), twin.py (evaluation twin), make_figures.py, requirements.txt, README.md
- results/ all tables as CSV, figures as JPG, metrics.json

License: MIT

Reproduce
    pip install -r code/requirements.txt   # plus Graphviz `dot`
    cd code && python3 run_all.py           # about 6.5 min on 2 CPU cores; regenerates results/ and ../figures/
