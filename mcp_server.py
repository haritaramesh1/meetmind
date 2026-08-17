from mcp.server.fastmcp import FastMCP
from memory import smart_search

mcp = FastMCP("meetmind")

@mcp.tool()
def search_meetings(question: str) -> str:
    """Search indexed meeting transcripts and slide text for relevant excerpts."""
    results = smart_search(question, k=3)
    if not results:
        return "No indexed meeting excerpts found."
    return "\n\n".join(
        f"[{source}]\n{text}" for text, source, _score in results
    )

if __name__ == "__main__":
    mcp.run()
