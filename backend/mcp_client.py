"""Example MCP client for testing the APIWeaver server."""

from typing import Dict, Any
import json


class MCPClient:
    """Example client for interacting with MCP server."""
    
    def __init__(self, server_name: str = "TCSBAnCSAPIWeaver"):
        self.server_name = server_name
        print(f"MCP Client connected to: {server_name}")
    
    async def register_bancs_api(self, base_url: str):
        """Register BAnCS API with provided URL."""
        config = {
            "name": "BAnCS",
            "base_url": base_url,
            "description": "TCS BAnCS Banking API",
            "auth": {
                "type": "custom",
                "custom_headers": {
                    "entity": "GPRDTTSTOU",
                    "userId": "1",
                    "languageCode": "1"
                }
            },
            "endpoints": [
                {
                    "name": "get_account_balance",
                    "method": "GET",
                    "path": "/Core/accountManagement/account/balanceDetails/{accountReference}",
                    "description": "Get account balance details",
                    "params": [
                        {
                            "name": "accountReference",
                            "type": "string",
                            "location": "path",
                            "required": True,
                            "description": "15-digit account reference number"
                        }
                    ]
                },
                {
                    "name": "get_customer_details",
                    "method": "GET",
                    "path": "/Core/customerManagement/customer/viewDetails",
                    "description": "Get customer details",
                    "params": [
                        {
                            "name": "CustomerID",
                            "type": "string",
                            "location": "query",
                            "required": True,
                            "description": "Customer ID"
                        },
                        {
                            "name": "pageNum",
                            "type": "integer",
                            "location": "query",
                            "required": False,
                            "default": 1
                        },
                        {
                            "name": "pageSize",
                            "type": "integer",
                            "location": "query",
                            "required": False,
                            "default": 22
                        }
                    ]
                },
                {
                    "name": "create_account",
                    "method": "POST",
                    "path": "/Core/accountManagement/account",
                    "description": "Create a new account",
                    "params": [
                        {
                            "name": "customerId",
                            "type": "integer",
                            "location": "body",
                            "required": True,
                            "description": "Customer ID"
                        },
                        {
                            "name": "accountName",
                            "type": "string",
                            "location": "body",
                            "required": True,
                            "description": "Account name"
                        },
                        {
                            "name": "accountType",
                            "type": "integer",
                            "location": "body",
                            "required": True,
                            "description": "Account type (1=Savings, 2=Current)"
                        },
                        {
                            "name": "accountCurrency",
                            "type": "string",
                            "location": "body",
                            "required": False,
                            "default": "USD",
                            "description": "Currency code"
                        },
                        {
                            "name": "branchId",
                            "type": "string",
                            "location": "body",
                            "required": True,
                            "description": "Branch ID"
                        },
                        {
                            "name": "product",
                            "type": "integer",
                            "location": "body",
                            "required": True,
                            "description": "Product code"
                        }
                    ]
                }
            ]
        }
        
        print(f"Registering BAnCS API at {base_url}")
        print(f"Configuration: {json.dumps(config, indent=2)}")
        
        # In real implementation, this would call the MCP server
        return {"success": True, "message": "API registered successfully"}
    
    async def call_api(self, api_name: str, endpoint_name: str, parameters: Dict[str, Any]):
        """Call a registered API endpoint."""
        print(f"Calling {api_name}.{endpoint_name}")
        print(f"Parameters: {parameters}")
        
        # In real implementation, this would call the MCP server
        return {"success": True, "data": {"example": "response"}}


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test():
        client = MCPClient()
        
        # Register API with custom URL
        base_url = input("Enter BAnCS API base URL (e.g., https://demoapps.tcsbancs.com): ")
        await client.register_bancs_api(base_url)
        
        # Call an endpoint
        result = await client.call_api(
            "BAnCS",
            "get_account_balance",
            {"accountReference": "101000000101814"}
        )
        print(f"Result: {result}")
    
    asyncio.run(test())