# BlueCity node sampling

## Requirements

* `uv`
* `lausanne_drive.graphml` network. This network can be created from the [BlueCity viz prototype repository](https://github.com/EPFL-ENAC/bluecity-viz), in the `dev` branch, by running `cd processing/traffic-analysis && make network`


## Installing the dependencies 

```bash
uv sync
```

## Running the script

```bash
uv run python3 bluecity_node_sampling.py
```
