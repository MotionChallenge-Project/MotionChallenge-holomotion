# Motion Challenge · holomotion

[holomotion](https://github.com/HorizonRobotics/HoloMotion) checkpoint adapter. G1 29DoF; LAFAN40 is the first suite.

## Install and run

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-holomotion.git
cd motion-challenge-holomotion
uv sync
uv run evaluate.py
```

Reuses local `checkpoints/policy.onnx` or the external cache; missing artifacts are
fetched and hash-checked. Use `--weights PATH --data PATH --offline` for local files.
Results and traces are written outside Git. `--help` lists options.

The root `motion-challenge` submodule owns data, physics, metrics and verification.
This repository only adapts model inference; no `sim2real` dependency.

[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.


## Checkpoint source

[HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion/blob/fcb8d4d8e96af9535c2adcc2b53e267019f865a2/README.md) · [model_14000.onnx](https://huggingface.co/HorizonRobotics/HoloMotion_models/blob/e3c90dd1f7483d02d3b4d2e5fb3a359c6ff9c67a/HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx). Author-released checkpoint.

The adapter pins this published artifact by SHA256. Artifact identity is separate from inference parity.

One `uv run evaluate.py` produces one verified `result.json`: success, completion,
three MPBPE alignments through first failure and the same three over the full motion.
Both scopes use the same checkpoint and source revision; no branch switching is needed.
