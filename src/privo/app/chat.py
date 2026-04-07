from rich.console import Console
from rich.panel import Panel
from rich.align import Align


class Chat:
    def __init__(self, console: Console) -> None:
        self.console = console

    def print_chat(self, text: str, name: str, align: str = "left") -> None:
        bubble = Panel(
            text,
            title=f"[bold blue]{name}[/bold blue]",
            border_style="bright_black",
            padding=(1, 2),
            expand=False,
            width=80,
        )
        self.console.print("")
        self.console.print(
            Align(
                bubble, align=align if align in ("left", "center", "right") else "left"
            )
        )
