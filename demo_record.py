import os
from dotenv import load_dotenv
load_dotenv()

from src.agent import agent

print("=== ASTER & ROW SUPPORT AGENT DEMO ===\n")

queries = [
    ("Standard Return Window", "What's the return window for regular customers?"),
    ("TrailPlus Return Window", "I have TrailPlus membership - what's my return window?"),
    ("Order Lookup", "Check my order ORD-1001"),
    ("International Shipping", "Can you ship to Germany?"),
    ("Handoff Trigger", "Are all your bag materials vegan?"),
]

for title, query in queries:
    print(f"\n{'='*60}")
    print(f"📝 {title}")
    print(f"👤 USER: {query}")
    print(f"{'='*60}")
    r = agent.process_message(f"demo-{title}", query)
    print(f"🤖 AGENT: {r.answer[:500]}...")
    if r.tool_calls:
        print(f"🔧 TOOLS: {[c['name'] for c in r.tool_calls]}")
    if r.handoff_recommended:
        print("🤝 HANDOFF RECOMMENDED")
    if r.sources:
        print(f"📚 SOURCES: {r.sources[:3]}")