import sys

from mcp.server.fastmcp import FastMCP

from memory import smart_search, start_model_loading


mcp = FastMCP("meetmind")


@mcp.tool()
def search_meetings(question: str) -> str:
    """Search MeetMind's indexed meeting memory."""

    try:
        results = smart_search(question, k=5)

        if not results:
            return (
                "No relevant indexed meeting evidence was found "
                f"for: {question}"
            )

        output = [
            f"MeetMind meeting evidence for: {question}",
            "",
            f"Found {len(results)} relevant excerpts:",
            "",
        ]

        for number, (text, source, score) in enumerate(
            results,
            start=1,
        ):
            output.append(
                f"--- Excerpt {number} ---"
            )

            if isinstance(text, dict):
                output.append(
                    f"Meeting: {text.get('source', source)}"
                )
                output.append(
                    f"Time: {text.get('start', '?')}s - "
                    f"{text.get('end', '?')}s"
                )
                output.append(
                    f"Speaker: {text.get('speaker', 'UNKNOWN')}"
                )
                output.append(
                    f"Relevance: {score:.3f}"
                )
                output.append(
                    f"Evidence: {text.get('text', '')}"
                )
            else:
                output.append(
                    f"Source: {source}"
                )
                output.append(
                    f"Relevance: {score:.3f}"
                )
                output.append(
                    f"Evidence: {text}"
                )

            output.append("")

        return "\n".join(output)

    except Exception as exc:
        print(
            f"search_meetings error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

        return (
            f"MeetMind search error: "
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":

    print(
        "MeetMind MCP server starting...",
        file=sys.stderr,
        flush=True,
    )

    # Start the model in the background.
    # DO NOT wait for it during MCP startup.
    start_model_loading()

    print(
        "MeetMind MCP server ready.",
        file=sys.stderr,
        flush=True,
    )

    mcp.run()