from dataclasses import dataclass, field


@dataclass
class ATSCArguments:
    total_steps: int = field(default=1000000, metadata={'help': "The total number of sampling steps during training."})
    train_steps: int = field(default=1000, metadata={'help': "The interval (steps) of training."})
    gamma: float = field(default=0.95, metadata={'help': "The discount factor."})
