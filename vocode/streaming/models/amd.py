from vocode.streaming.models.model import BaseModel


class AMDConfig(BaseModel):
    enabled: bool
    callback_url: str
    threshold: float = 5
    keywords: list[str] = []
