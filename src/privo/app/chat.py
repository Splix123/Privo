from rich.console import Console
from rich.panel import Panel
from rich.align import Align


class Chat:
    """Klasse um Chatbubbles zu generieren"""

    def __init__(self, console: Console) -> None:
        """Nimmt ein Konsolenobjekt keine redundanten Konsolen zu eröffnen und setzt die maximale Chatbubble-Breite auf 80 Zeichen.

        Args:
            console (Console): Rich Console Objekt
        """
        self.console = console
        self.width = 80

    def print_chat(self, text: str, name: str, align: str = "left") -> None:
        """Gibt eine Chatbubble in der Konsole aus.

        Args:
            text (str): Der Text der Chatbubble.
            name (str): Der Name des Absenders.
            align (str, optional): Die Ausrichtung der Chatbubble ("left", "center", "right"). Defaults to "left".
        """
        if not text.strip():
            return

        if align not in ("left", "center", "right"):
            align = "left"

        bubble = Panel(
            text,
            title=f"[bold blue]{name}[/bold blue]",
            border_style="bright_black",
            padding=(1, 2),
            expand=False,
            width=self.width,
        )
        self.console.print()
        self.console.print(Align(bubble, align))
