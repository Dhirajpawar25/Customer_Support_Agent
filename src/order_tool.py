"""Order lookup tool for the Aster & Row AI Support Agent."""
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.models import Order, ToolCall
from src.config import config


class OrderLookup:
    """Order lookup tool that safely retrieves order information."""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self._load_orders()
    
    def _load_orders(self) -> None:
        """Load orders from JSON file."""
        with open(config.orders_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for order_data in data.get("orders", []):
            order = Order.from_dict(order_data)
            self.orders[order.order_id.upper()] = order
    
    def normalize_order_id(self, order_id: str) -> str:
        """Normalize order ID - uppercase, strip whitespace."""
        return order_id.strip().upper()
    
    def lookup(self, order_id: str) -> ToolCall:
        """Look up an order by ID. Returns a ToolCall with result or error."""
        normalized_id = self.normalize_order_id(order_id)
        
        tool_call = ToolCall(
            name="order_lookup",
            arguments={"order_id": order_id}
        )
        
        if not normalized_id:
            tool_call.error = "Order ID is required"
            tool_call.result = {"error": "Order ID is required", "success": False}
            return tool_call
        
        # Validate format (ORD-XXXX)
        if not re.match(r'^ORD-\d+$', normalized_id):
            tool_call.error = f"Invalid order ID format: {order_id}. Expected format: ORD-XXXX"
            tool_call.result = {
                "error": f"Invalid order ID format: {order_id}. Expected format: ORD-XXXX",
                "success": False
            }
            return tool_call
        
        order = self.orders.get(normalized_id)
        
        if not order:
            tool_call.error = f"Order {normalized_id} not found"
            tool_call.result = {
                "error": f"Order {normalized_id} not found",
                "success": False,
                "order_id": normalized_id
            }
            return tool_call
        
        # Return safe order data
        safe_data = order.to_safe_dict()
        tool_call.result = {
            "success": True,
            "order": safe_data
        }
        return tool_call
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get full order object (for internal use)."""
        normalized_id = self.normalize_order_id(order_id)
        return self.orders.get(normalized_id)
    
    def list_orders(self) -> List[str]:
        """List all order IDs."""
        return sorted(self.orders.keys())


# Global instance
order_lookup = OrderLookup()