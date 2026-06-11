# PNA Baseline for AML Subgraph Classification

This repo contains a standalone Principal Neighbourhood Aggregation (PNA)
baseline for the HST AML/SAML-D subgraph-level task.

The loader consumes the AML-normalized builder outputs:

- `background_nodes.csv`
- `background_edges.csv`
- `nodes.csv`
- `component_edges.csv`
- `connected_components.csv`

Each connected component/subgraph becomes one PyG `Data` graph. Node IDs are
relabelled locally, directed parallel transaction edges are preserved, and
`feat*` transaction columns are used as edge attributes by default.

## Run

```bash
conda activate HST
cd /home/ghonkoop/repos/PNA
python train.py --config configs/aml/HIS.yaml
```

SLURM:

```bash
sbatch run/run_train.sh
sbatch --export=ALL,CONFIG=configs/aml/LIS.yaml run/run_train.sh
sbatch run/run_optuna.sh
```

## Notes

This is a direct graph-level classifier baseline. It does not build HST's
subgraph-supergraph. It tests whether a strong edge-feature-aware graph GNN on
the original component graphs is competitive with the hierarchical method.
