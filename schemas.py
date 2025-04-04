from pydantic import BaseModel, Field
from typing import List, Optional


class WebhookMetadata(BaseModel):
    display_phone_number: str
    phone_number_id: str


class WebhookMessageText(BaseModel):
    body: str


class WebhookAudio(BaseModel):
    mime_type: str
    sha256: str
    id: str
    voice: bool


class WebhookMessage(BaseModel):
    from_: str = Field(..., alias="from")
    id: str
    timestamp: str
    type: str
    text: Optional[WebhookMessageText] = None
    audio: Optional[WebhookAudio] = None


class WebhookValue(BaseModel):
    messaging_product: str
    metadata: WebhookMetadata
    contacts: List[dict]
    messages: List[WebhookMessage]


class WebhookChange(BaseModel):
    field: str
    value: WebhookValue


class WebhookEntry(BaseModel):
    id: str
    changes: List[WebhookChange]


class WebhookPayload(BaseModel):
    object: str
    entry: List[WebhookEntry]
