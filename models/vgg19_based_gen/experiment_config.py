import dataclasses


@dataclasses.dataclass
class ExperimentConfig:
    alpha: float
    beta: float
    use_upsampling: bool
    use_reflection_padding: bool
    lr: float
    num_epochs: int