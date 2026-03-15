from pydantic import BaseModel,Field
from typing import Dict, List, Any

class PriorityList(BaseModel):
    tool_priority_list: List[Dict[str, Any]] = Field(..., description="List of tools in the order they should be applied")
    
    
    
