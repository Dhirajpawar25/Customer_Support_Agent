"""Main AI Support Agent for Aster & Row."""
import json
import re
from typing import List, Optional, Dict, Any
from openai import OpenAI
from src.config import config
from src.models import (
    AgentResponse, ConversationSession, ConversationTurn,
    DocumentChunk, ToolCall, RetrievalResult
)
from src.embeddings import embedding_store
from src.order_tool import order_lookup
from src.document_loader import load_knowledge_base, filter_authoritative_chunks


class AsterRowAgent:
    """AI Support Agent for Aster & Row."""
    
    def __init__(self):
        self._model = None
        self.sessions: Dict[str, ConversationSession] = {}
        self._kb_initialized = False
    
    @property
    def model(self):
        """Lazy initialization of the configured OpenAI-compatible client."""
        if self._model is None:
            self._model = OpenAI(
                api_key=config.llm_api_key or config.openai_api_key,
                base_url=config.llm_base_url
            )
        return self._model
    
    def _initialize_knowledge_base(self) -> None:
        if self._kb_initialized:
            return
        all_chunks = load_knowledge_base()
        embedding_store.build_index(all_chunks)
        if config.debug_mode:
            print(f"Loaded {len(all_chunks)} chunks from knowledge base")
            auth_chunks = filter_authoritative_chunks(all_chunks)
            print(f"Authoritative chunks: {len(auth_chunks)}")
        self._kb_initialized = True
    
    def get_session(self, session_id: str) -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        return self.sessions[session_id]

    def _build_system_prompt(self) -> str:
        return """You are the Aster & Row AI Support Agent. You help customers with questions about policies, products, and orders.

## Core Instructions

1. **Use only company content** - Answer using the retrieved knowledge base passages and order lookup tool. Do not use general knowledge for company-specific questions.

2. **Cite sources** - Every policy or product answer must include source references (filename and heading). Format: "Source: 01-returns-policy-current.md > Standard return window"

3. **Prefer authoritative sources** - Active, official, customer-facing documents take precedence. Never cite superseded (02-returns-policy-legacy.md) or internal (14-internal-content-migration-notes.md) documents as authority.

4. **Handle conflicts explicitly** - If current authoritative sources conflict, surface the conflict rather than silently choosing one. Recommend human assistance.

5. **Safe abstention** - If information is insufficient, clearly say so and recommend human help if appropriate. Do not invent answers.

6. **Order lookup** - Use the order_lookup tool when customers ask about specific orders. Never expose internal fields (email, address, risk score, internal notes, support tags).

7. **Privacy** - Never disclose customer PII (email, address), internal notes, risk scores, or internal-only fields. If asked, refuse and recommend human assistance.

8. **No false claims** - Never claim a lookup happened when it didn't. Never promise refunds, cancellations, replacements, or address changes unless the system actually supports that action.

9. **Multi-turn context** - Maintain conversation context. Follow-ups like "What about Canada?" should use previous context.

10. **Handoff** - Recommend human assistance when:
    - Documents conflict
    - Information is insufficient
    - Customer requests action you cannot complete
    - Privacy-sensitive data is requested

## Response Format

Provide clear, helpful answers. Include:
- The answer
- Sources (when applicable)
- Whether human handoff is recommended

Be concise but complete."""

    def _build_retrieval_context(self, retrieval: RetrievalResult) -> str:
        if not retrieval.chunks:
            return "No relevant policy documents found."
        
        context_parts = []
        for i, (chunk, score) in enumerate(zip(retrieval.chunks, retrieval.scores)):
            citation = chunk.get_citation()
            context_parts.append(f"[Source {i+1}: {citation} (relevance: {score:.2f})]\n{chunk.content}")
        
        return "\n\n---\n\n".join(context_parts)

    def _detect_order_id(self, message: str) -> Optional[str]:
        match = re.search(r'ORD-\d+', message, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        return None
    
    def _should_lookup_order(self, message: str, history: List[Dict[str, str]]) -> bool:
        if self._detect_order_id(message):
            return True
        
        for turn in reversed(history[-3:]):
            if turn["role"] == "assistant" and "ORD-" in turn["content"]:
                followup_keywords = ["when", "where", "arrive", "deliver", "ship", "track", "status", "eta", "estimate"]
                if any(kw in message.lower() for kw in followup_keywords):
                    return True
        
        return False
    
    def _extract_order_id_from_context(self, history: List[Dict[str, str]]) -> Optional[str]:
        for turn in reversed(history[-5:]):
            if turn["role"] == "assistant":
                match = re.search(r'ORD-\d+', turn["content"])
                if match:
                    return match.group(0).upper()
        return None
    
    def _call_llm(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Any:
        """Call the configured OpenAI-compatible chat completion endpoint."""
        response = self.model.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=config.llm_temperature,
            tools=tools or None,
            timeout=60,
        )
        message = response.choices[0].message
        function_calls = []
        for tool_call in message.tool_calls or []:
            function_calls.append(type("FunctionCall", (), {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            })())
        message.function_calls = function_calls
        return response
    
    def _format_tool_result(self, tool_call: ToolCall) -> str:
        if tool_call.error:
            return f"Error: {tool_call.error}"
        return json.dumps(tool_call.result, indent=2)

    def process_message(self, session_id: str, user_message: str) -> AgentResponse:
        # Initialize knowledge base on first use
        self._initialize_knowledge_base()
        
        session = self.get_session(session_id)
        
        session.add_turn("user", user_message)
        history = session.get_history()
        
        tool_calls = []
        order_context = ""
        
        if self._should_lookup_order(user_message, history):
            order_id = self._detect_order_id(user_message)
            if not order_id:
                order_id = self._extract_order_id_from_context(history)
            
            if order_id:
                tool_call = order_lookup.lookup(order_id)
                tool_calls.append(tool_call)
                
                if tool_call.result and tool_call.result.get("success"):
                    order_data = tool_call.result["order"]
                    order_context = f"\n\nOrder Information (ORD-{order_id}):\n{json.dumps(order_data, indent=2)}"
                else:
                    order_context = f"\n\nOrder Lookup Result: {tool_call.error}"
        
        retrieval = embedding_store.search_authoritative(
            user_message, 
            top_k=config.top_k, 
            min_score=config.min_score_threshold
        )
        
        retrieval_context = self._build_retrieval_context(retrieval)
        
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        
        for turn in history[:-1]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        
        current_content = user_message
        if order_context:
            current_content += order_context
        if retrieval_context:
            current_content += f"\n\nRelevant Policy Information:\n{retrieval_context}"
        
        messages.append({"role": "user", "content": current_content})
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "order_lookup",
                    "description": "Look up order information by order ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "Order ID in format ORD-XXXX"
                            }
                        },
                        "required": ["order_id"]
                    }
                }
            }
        ]
        
        response = self._call_llm(messages, tools)
        message = response.choices[0].message
        
        if message.function_calls:
            for func_call in message.function_calls:
                if func_call.name == "order_lookup":
                    args = json.loads(func_call.arguments)
                    order_id = args.get("order_id", "")
                    result_call = order_lookup.lookup(order_id)
                    tool_calls.append(result_call)
                    
                    messages.append({
                        "role": "assistant",
                        "content": f"Function call: {func_call.name}({func_call.arguments})"
                    })
                    messages.append({
                        "role": "user",
                        "content": self._format_tool_result(result_call)
                    })
            
            response = self._call_llm(messages, tools)
            message = response.choices[0].message
        
        answer = message.content or ""
        
        handoff_recommended = False
        handoff_reason = None
        
        handoff_keywords = [
            "human assistance", "human review", "contact support", 
            "escalate to human", "speak with a human", "human agent"
        ]
        if any(kw in answer.lower() for kw in handoff_keywords):
            handoff_recommended = True
            handoff_reason = "Agent recommended human assistance"
        
        source_files = set()
        for chunk in retrieval.chunks:
            source_files.add(chunk.metadata.source_file)
        
        conflict_pairs = [
            ("11-product-care.md", "12-breeze-tumbler-product-card.md")
        ]
        for f1, f2 in conflict_pairs:
            if f1 in source_files and f2 in source_files:
                handoff_recommended = True
                handoff_reason = f"Conflicting sources detected: {f1} and {f2}"
        
        agent_response = AgentResponse(
            answer=answer,
            sources=[c.get_citation() for c in retrieval.chunks],
            handoff_recommended=handoff_recommended,
            handoff_reason=handoff_reason,
            tool_calls=tool_calls,
            retrieved_chunks=retrieval.chunks,
            retrieval_scores=retrieval.scores,
            debug_info={
                "session_id": session_id,
                "user_message": user_message,
                "history_length": len(history),
                "retrieval_count": len(retrieval.chunks),
                "tool_calls_count": len(tool_calls)
            }
        )
        
        session.add_turn("assistant", answer, agent_response)
        
        return agent_response


agent = AsterRowAgent()