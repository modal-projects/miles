# Modal repository-repair rollouts

This adapter runs the standard mini-swe-agent control loop inside the Miles
rollout worker, so every model request passes through Miles' TITO session
server. Repository commands and a task-provided verifier execute inside one
fresh Modal Sandbox per attempt.

Each prompt identifies a task directory containing an environment Dockerfile,
an optional pre-agent `environment/setup.sh`, and a `tests/test.sh` verifier.
The adapter creates a sandbox from the Dockerfile base image, blocks network
access, runs setup before the policy, and uploads verifier assets only after
policy interaction ends. Dataset preparation and splitting belong in
benchmark-specific experiment configuration.

This directory does not contain a launcher. Use the corresponding configuration
in `multinode-training-guide/miles/configs`. Preparing data or checkpoints does
not start training; do not invoke the training entrypoint without approval.
