from dataclasses import dataclass

@dataclass
class Candidate:
    kind: str
    source_id: int
    target_id: int
    angle: float
    ships: int
    eta: float
    score: float


@dataclass
class MultiCandidate:
    kind: str
    target_id: int
    orders: list[Candidate]
    eta: float
    score: float
