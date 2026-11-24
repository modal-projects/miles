from miles.utils.args.schema import A, Arg, BaseConfig


class TinkerConfig(BaseConfig):
    tinker_lora_groups: list[str] | None = None

    tinker_server_host: A[str, Arg()] = "0.0.0.0"
    tinker_server_port: A[int, Arg()] = 10613
    tinker_base_model: A[
        str | None,
        Arg(help="Model name advertised by the gateway (default: --hf-checkpoint)"),
    ] = None
    tinker_checkpoint_root: A[
        str | None,
        Arg(help="Directory for tinker:// checkpoints (required by serve_tinker.py)"),
    ] = None
    tinker_public_url: A[
        str | None,
        Arg(
            help="URL clients reach the gateway at, e.g. https://gw.example.com/tinker/ behind a TLS or path-prefix "
            "ingress; download URLs are built from it (default: the URL each request arrived at)"
        ),
    ] = None
    tinker_sampler_root: A[
        str | None,
        Arg(
            help="Directory for sampler saves, the adapters the engines serve, apart from training checkpoints and "
            "downloads (default: --tinker-checkpoint-root)"
        ),
    ] = None
    tinker_sampler_backend: A[
        str,
        Arg(
            choices=["snapshot", "in_place"],
            help="snapshot (Tinker's contract): a sampler path serves the weights saved at it. in_place: each "
            "model's saves replace its one adapter on every engine, and every sampler path of the model serves its "
            "latest save",
        ),
    ] = "snapshot"
