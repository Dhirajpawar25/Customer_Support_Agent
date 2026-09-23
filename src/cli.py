"""CLI interface for the Aster & Row AI Support Agent."""
import uuid
import json
from src.agent import agent
from src.config import config


def print_response(response):
    """Print agent response in a readable format."""
    print("\n" + "="*60)
    print("ASSISTANT:")
    print(response.answer)
    
    if response.sources:
        print("\nSources:")
        for src in response.sources:
            print(f"  - {src}")
    
    if response.handoff_recommended:
        print(f"\n⚠️  HUMAN HANDOFF RECOMMENDED: {response.handoff_reason}")
    
    if response.tool_calls:
        print("\nTool Calls:")
        for tc in response.tool_calls:
            status = "✓" if not tc.error else "✗"
            print(f"  {status} {tc.name}({tc.arguments})")
            if tc.error:
                print(f"     Error: {tc.error}")
    
    if config.debug_mode:
        print("\n--- Debug Info ---")
        print(json.dumps(response.debug_info, indent=2))


def main():
    """Main CLI loop."""
    session_id = str(uuid.uuid4())[:8]
    print("Aster & Row AI Support Agent")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'debug' to toggle debug mode")
    print(f"Session ID: {session_id}")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break
        
        if user_input.lower() == 'debug':
            config.debug_mode = not config.debug_mode
            print(f"Debug mode: {'ON' if config.debug_mode else 'OFF'}")
            continue
        
        if not user_input:
            continue
        
        try:
            response = agent.process_message(session_id, user_input)
            print_response(response)
        except Exception as e:
            print(f"\nError: {e}")
            if config.debug_mode:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()