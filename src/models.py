"""Data models for the Aster & Row AI Support Agent."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentAuthority(str, Enum):
    """Document authority levels for precedence handling."""
    OFFICIAL = "official"
    INTERNAL = "internal"
    LEGACY = "legacy"


class DocumentStatus(str, Enum):
    """Document status for filtering."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DRAFT = "draft"


@dataclass
class DocumentMetadata:
    """Metadata for a policy document."""
    document_id: str
    title: str
    policy_authority: DocumentAuthority
    status: DocumentStatus
    tags: List[str] = field(default_factory=list)
    audience: str = "customer-facing"
    last_updated: str = ""
    source_file: str = ""
    effective_date: Optional[str] = None
    last_reviewed: Optional[str] = None
    supersedes: Optional[str] = None

    @property
    def is_authoritative(self) -> bool:
        """Check if document is authoritative for customer-facing answers."""
        return (
            self.status == DocumentStatus.ACTIVE
            and self.policy_authority == DocumentAuthority.OFFICIAL
            and self.audience == "customer-facing"
        )
    
    @property
    def is_superseded(self) -> bool:
        """Check if document is superseded."""
        return self.status == DocumentStatus.SUPERSEDED


@dataclass
class DocumentChunk:
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    heading: Optional[str] = None
    heading_level: int = 0
    embedding: Optional[List[float]] = None
    
    def get_citation(self) -> str:
        parts = [self.metadata.source_file]
        if self.heading:
            parts.append(self.heading)
        return " > ".join(parts)


@dataclass
class Order:
    order_id: str
    customer_name: str
    customer_email: str
    shipping_address: str
    membership_tier: str
    items: List[Dict[str, Any]]
    placed_at: str
    status: str
    status_updated_at: str
    shipped_at: Optional[str]
    delivered_at: Optional[str]
    carrier: Optional[str]
    tracking_number: Optional[str]
    estimated_delivery: Optional[str]
    customer_safe_message: str
    internal_risk_score: int
    internal_warehouse_note: str
    internal_support_tags: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        internal = data.get("internal", {})
        customer = data.get("customer", {})
        return cls(
            order_id=data["order_id"],
            customer_name=customer.get("name", ""),
            customer_email=customer.get("email", ""),
            shipping_address=customer.get("shipping_address", ""),
            membership_tier=data.get("membership_tier", "standard"),
            items=data.get("items", []),
            placed_at=data.get("placed_at", ""),
            status=data.get("status", ""),
            status_updated_at=data.get("status_updated_at", ""),
            shipped_at=data.get("shipped_at"),
            delivered_at=data.get("delivered_at"),
            carrier=data.get("carrier"),
            tracking_number=data.get("tracking_number"),
            estimated_delivery=data.get("estimated_delivery"),
            customer_safe_message=data.get("customer_safe_message", ""),
            internal_risk_score=internal.get("risk_score", 0),
            internal_warehouse_note=internal.get("warehouse_note", ""),
            internal_support_tags=internal.get("support_tags", []),
        )
    
    def to_safe_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "customer_name": self.customer_name,
            "membership_tier": self.membership_tier,
            "items": [
                {
                    "sku": item.get("sku"),
                    "name": item.get("name"),
                    "quantity": item.get("quantity"),
                    "final_sale": item.get("final_sale", False)
                }
                for item in self.items
            ],
            "placed_at": self.placed_at,
            "status": self.status,
            "status_updated_at": self.status_updated_at,
            "shipped_at": self.shipped_at,
            "delivered_at": self.delivered_at,
            "carrier": self.carrier,
            "tracking_number": self.tracking_number,
            "estimated_delivery": self.estimated_delivery,
            "customer_safe_message": self.customer_safe_message,
        }
@dataclass
class RetrievalResult:
    chunks: List['DocumentChunk']
    query: str
    scores: List[float]


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class AgentResponse:
    answer: str
    sources: List[str] = field(default_factory=list)
    handoff_recommended: bool = False
    handoff_reason: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    retrieved_chunks: List['DocumentChunk'] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "handoff_recommended": self.handoff_recommended,
            "handoff_reason": self.handoff_reason,
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "error": tc.error
                }
                for tc in self.tool_calls
            ],
            "retrieved_chunks": [
                {
                    "content": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                    "citation": c.get_citation(),
                    "metadata": {
                        "document_id": c.metadata.document_id,
                        "title": c.metadata.title,
                        "authority": c.metadata.policy_authority.value,
                        "status": c.metadata.status.value
                    }
                }
                for c in self.retrieved_chunks
            ],
            "retrieval_scores": self.retrieval_scores,
            "debug_info": self.debug_info
        }


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_response: Optional[AgentResponse] = None


@dataclass
class ConversationSession:
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_turn(self, role: str, content: str, agent_response: Optional[AgentResponse] = None):
        turn = ConversationTurn(role=role, content=content, agent_response=agent_response)
        self.turns.append(turn)
        self.updated_at = datetime.now().isoformat()
    
    def get_history(self, max_turns: int = 10) -> List[Dict[str, str]]:
        history = []
        for turn in self.turns[-max_turns:]:
            history.append({"role": turn.role, "content": turn.content})
        return history
    
    def get_last_user_message(self) -> Optional[str]:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None