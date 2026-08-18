from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-server")

@mcp.tool()
def hello() -> str:
    """Simple MCP connection test."""
    return "MCP connection works!"

if __name__ == "__main__":
    mcp.run()