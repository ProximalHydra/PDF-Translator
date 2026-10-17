from dataclasses import dataclass, field


class DocumentError(ValueError):
    pass


class TranslationError(RuntimeError):
    pass


@dataclass
class Run:
    id: str
    text: str
    font: str
    size: float
    color: int
    flags: int
    join_before: str = ""


@dataclass
class Block:
    id: str
    page: int
    bbox: tuple[float, float, float, float]
    runs: list[Run]
    rotation: int = 0
    supported: bool = True
    alignment: str = "left"


@dataclass
class Document:
    pages: int
    blocks: list[Block]
    characters: int

    @property
    def runs(self):
        return [run for block in self.blocks for run in block.runs]


@dataclass
class Issue:
    code: str
    message: str
    page: int | None = None
    block_id: str | None = None

    def as_dict(self):
        return {"code": self.code, "message": self.message,
                "page": self.page, "block_id": self.block_id}


@dataclass
class Review:
    passed: int = 0
    corrected: int = 0
    issues: list[Issue] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)
