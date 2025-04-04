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


# New models for statuses
class WebhookConversationOrigin(BaseModel):
    type: str


class WebhookConversation(BaseModel):
    id: str
    origin: WebhookConversationOrigin


class WebhookPricing(BaseModel):
    billable: bool
    pricing_model: str
    category: str


class WebhookStatus(BaseModel):
    id: str
    status: str
    timestamp: str
    recipient_id: str
    conversation: Optional[WebhookConversation] = None
    pricing: Optional[WebhookPricing] = None


class WebhookValue(BaseModel):
    messaging_product: str
    metadata: WebhookMetadata
    contacts: Optional[List[dict]] = None
    messages: Optional[List[WebhookMessage]] = None
    statuses: Optional[List[WebhookStatus]] = None


class WebhookChange(BaseModel):
    field: str
    value: WebhookValue


class WebhookEntry(BaseModel):
    id: str
    changes: List[WebhookChange]


class WebhookPayload(BaseModel):
    object: str
    entry: List[WebhookEntry]
