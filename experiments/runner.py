import dataclasses


@dataclasses.dataclass
class ExperimentConfig:
    alpha: float
    beta: float
    use_upsampling: bool
    use_reflection_padding: bool
    lr: float
    num_epochs: int
    




EXPERIMENT_CONFIGS = [
    ExperimentConfig(
        alpha=1e5,
        beta=1e1,
        use_upsampling=True,
        use_reflection_padding=True,
        lr=1e-4,
        num_epochs=3,
    ),
    ExperimentConfig(
        alpha=1e4,
        beta=1e2,
        use_upsampling=True,
        use_reflection_padding=True,
        lr=5e-4,
        num_epochs=1,
    ),
    ExperimentConfig(
        alpha=1e3,
        beta=1e3,
        use_upsampling=True,
        use_reflection_padding=True,
        lr=1e-4,
        num_epochs=3,
    ),
    ExperimentConfig(
        alpha=1e4,
        beta=1e2,
        use_upsampling=False,
        use_reflection_padding=False,
        lr=1e-4,
        num_epochs=2,
    ),
    ExperimentConfig(
        alpha=1e4,
        beta=1e2,
        use_upsampling=True,
        use_reflection_padding=True,
        lr=5e-5,
        num_epochs=5,
    ),
]
