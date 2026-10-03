"""Keep the experiment package out of SGLang's normal installation path."""


def install() -> None:
    from tests.manual.score_centering_spec.capture import install as install_hooks

    install_hooks()
