# A Data-Driven Study of the Mouse Medial Entorhinal Cortex
This repository contains the code I developed for analysing Neuropixels 2.0 recordings from the Nolan lab.

Contact information:
| Name         | Email address                                         | Website                   |
| :----------- | :---------------------------------------------------- | :------------------------ |
| Wolf De Wulf | [wolf.de.wulf@ed.ac.uk](mailto:wolf.de.wulf@ed.ac.uk) | https://www.wolfdewulf.eu |

## Usage
This is all linux-based.

### Environment

I use [`uv`](https://docs.astral.sh/uv/) to keep track of packages and versions.  
To recreate my exact environment, do:
```
uv sync
```

The [`uv.lock`](uv.lock) file contains a more detailed description of the environment, if you need it.

### Data

Read [`data/README.md`](data/README.md) for directions on where to get what data and how to preprocess it.

### Analyses

Various analysis scripts can be found in [`scripts/`](scripts/).  
The source code used in these can be found in [`src/spatial_manifolds/`](src/spatial_manifolds/).

## Rebuilding the AnchorDynamics2026 paper

The figures, the PDF mock-ups and the manuscript text of *AnchorDynamics2026*
all live in [`scripts/figures/AnchorDynamics2026/`](scripts/figures/AnchorDynamics2026/).
Start with [`README.md`](scripts/figures/AnchorDynamics2026/README.md) in that
folder: where the processed data is on the DataStore, how to point the scripts
at it, the command for each figure one at a time or all at once
(`rebuild_paper.py figures`), which notebooks are still live, and how to
rebuild the manuscript PDFs, which needs no data at all.
[`REBUILD.md`](scripts/figures/AnchorDynamics2026/REBUILD.md) goes deeper:
recomputing every intermediate result from the raw recordings
(`rebuild_paper.py derive`), the caches to clear first, what is deliberately
not in git, and troubleshooting.

## Questions

I am happy to answer your questions or to discuss my work, please send me an email!
