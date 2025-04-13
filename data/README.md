# Data Formatting
This documents presents instructions on where to get and put data for using this package.  
The main goal will be to format the raw data into `.nwb` files that are easy to work with using [`pynapple`]().

## Eddie
The raw data files are large, and there are many of them, so the main advice is to run this preprocessing step on [Eddie](https://www.wiki.ed.ac.uk/pages/viewpage.action?spaceKey=ResearchServices&title=Eddie).  
Automated scripts for doing the preprocessing can be found in [`HPC/preprocessing/`](HPC/preprocessing).  
The only thing you need to personalise is the path to your [SCRATCH](https://www.wiki.ed.ac.uk/display/ResearchServices/Storage) directory in [`all.sh`](HPC/preprocessing/all.sh):
```
storage="/exports/eddie/scratch/<eddie username here>"
```
After that, you can start the formatting tasks by calling that script on Eddie:
```
./HPC/preprocessing/all.sh
```
This will queue 3 tasks per mouse/day:
1. [`stagein.sh`](HPC/preprocessing/stagein.sh), which will copy the data for that experiment from the DataStore, to Eddie
2. [`preprocess.sh`](HPC/preprocessing/preprocess.sh), which will perform the actual preprocessing
3. [`cleanup.sh`](HPC/preprocessing/cleanup.sh), which will remove files afterwards

These tasks are queued in such a way that they run in sequence.  
Across mice and days, groups of tasks will run in parallel, but only 18 at a time, as we are limited by the storage in the SCRATCH directory (2TB).
Eddie will output logs for each group of tasks in a newly created `logs` folder.

The output of each task is stored in the `sessions` folder in your [SCRATCH](https://www.wiki.ed.ac.uk/display/ResearchServices/Storage) directory.  
The output consists of 3 folders, one for each session type: `OF1`, `VR`, and `OF2`.  
Each of those will contain a `M<mouse>D<day><session_type>.nwb` file as well as a `<sorter>.npz` file containing spike-sorted clusters and corresponding metrics (default sorter is `kilosort4`).
There will also be `sync/` folder containing plots about the applied synchronisation.

If you only want to preprocess a specific set of sessions, you can edit the hard-coded list of sessions in the [`all.sh`](HPC/preprocessing/all.sh) script:
```
sessions=(
    "20 14"
    "20 15"
    "20 16"
    ...
)
```

## Local
Sometimes you might want to preprocess sessions locally.  
For that, you'll again need to stage the data from the DataStore to your local device, which you can do using the same [`stagein.sh`](HPC/preprocessing/stagein.sh) script:
```
./HPC/preprocessing/stagein.sh <mouse> <day> <storage> <datastore>
```
Subsequently, you can manually preprocess the behaviour and clusters using the scripts in [`scripts/preprocessing`](scripts/preprocessing):
```
uv run scripts/preprocessing/behaviour.py --mouse <mouse> --day <day> --storage <storage> --dlc_model <deep lab cut model>
uv run scripts/preprocessing/sorting.py --mouse <mouse> --day <day> --storage <storage> --sorter <sorter>
```
