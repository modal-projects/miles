from pydantic import BaseModel, Field


class SessionRecord(BaseModel):
    timestamp: float
    request_timestamp: float | None = None
    method: str
    path: str
    # New session servers store only the prompt length.  The complete prompt
    # for each turn is a prefix of GetSessionResponse.metadata's single
    # accumulated_token_ids array, so repeating input_ids (and the growing
    # messages list) in every record makes long sessions quadratic on the wire.
    # request remains for backward compatibility with saved/older records.
    prompt_token_count: int | None = None
    request: dict = Field(default_factory=dict)
    response: dict
    status_code: int


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)
