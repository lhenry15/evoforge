"""Privacy configuration and rules."""

from __future__ import annotations

from pydantic import BaseModel


class PrivacyRules(BaseModel):
    pii_scrub: bool = True
    user_consent_required: bool = False
    redact_fields: list[str] = []         # field names to always redact
    sensitive_topics: list[str] = []      # topics routed only to authorized users

    @classmethod
    def strict(cls) -> "PrivacyRules":
        return cls(pii_scrub=True, user_consent_required=True)

    @classmethod
    def default(cls) -> "PrivacyRules":
        return cls()


class PrivacyConfig(BaseModel):
    rules: PrivacyRules = PrivacyRules()
