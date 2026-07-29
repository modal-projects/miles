# Modal SWE-Gym rollouts

This adapter runs the standard mini-swe-agent control loop inside the Miles
rollout worker, so every model request passes through Miles' TITO session
server. Repository commands and the canonical Harbor SWE-Gym verifier execute
inside one fresh Modal Sandbox per attempt.

Each prompt must identify a canonical Harbor `swegym` task directory. The
adapter creates a sandbox from that task's Dockerfile base image, blocks
network access, and uploads verifier tests only after policy interaction ends.
Dataset preparation and train/holdout splitting belong in the experiment
configuration.

This directory does not contain a launcher. Use the corresponding configuration
in `multinode-training-guide/miles/configs`. Preparing data or checkpoints does
not start training; do not invoke the training entrypoint without approval.
