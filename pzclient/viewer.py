"""
The window of the ``"human"`` render mode.

`Viewer` shows the PNG frames sent by the ``/render`` endpoint of the server in
a pygame window, as the local PettingZoo environments do in their ``"human"``
render mode.  pygame is an optional dependency of the library: it is only
needed by this render mode, and only imported when it is used.
"""

import io
import os
from types import ModuleType

# The color of the window around the frame, when their proportions differ
BACKGROUND_COLOR = (0, 0, 0)


def import_pygame() -> ModuleType:
    """
    Import pygame, which only the ``"human"`` render mode needs.

    Returns
    -------
    module
        The pygame module.

    Raises
    ------
    ImportError
        If pygame is not installed.
    """
    # Silence the banner pygame prints when it is imported
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    try:
        import pygame
    except ImportError as error:
        raise ImportError(
            "The 'human' render mode needs pygame: install it with "
            "'pip install pygame', or use render_mode='rgb_array'."
        ) from error

    return pygame


class Viewer:
    """
    A pygame window showing the frames of a remote environment.

    The window opens with the first frame, at the size of that frame.  It can
    be resized: the frames are then scaled to fit it, keeping their proportions.

    Parameters
    ----------
    caption : str
        The title of the window.

    Attributes
    ----------
    is_open : bool
        Whether the window is open.

    Raises
    ------
    ImportError
        If pygame is not installed.
    """

    def __init__(self, caption: str):
        self.caption = caption
        self.is_open = False
        self._pygame = import_pygame()

    def show(self, png: bytes) -> bool:
        """
        Show a frame in the window, opening the window if needed.

        Parameters
        ----------
        png : bytes
            The PNG encoded frame.

        Returns
        -------
        bool
            ``False`` if the user has closed the window, in which case the
            frame is not shown, ``True`` otherwise.
        """
        pygame = self._pygame

        if self.is_open:
            # Handle the events of the window, which also keeps the system
            # from deeming the program unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return False

        image = pygame.image.load(io.BytesIO(png), "frame.png")

        if not self.is_open:
            pygame.display.init()
            pygame.display.set_mode(image.get_size(), pygame.RESIZABLE)
            pygame.display.set_caption(self.caption)
            self.is_open = True

        # Fit the frame to the window, which the user may have resized (the
        # surface of the window is then replaced by a new one)
        window = pygame.display.get_surface()
        window_width, window_height = window.get_size()
        scale = min(window_width / image.get_width(), window_height / image.get_height())

        if scale != 1.0:
            image = pygame.transform.smoothscale(
                image,
                (
                    max(1, round(image.get_width() * scale)),
                    max(1, round(image.get_height() * scale)),
                ),
            )

        window.fill(BACKGROUND_COLOR)
        window.blit(image, image.get_rect(center=window.get_rect().center))
        pygame.display.flip()

        return True

    def close(self) -> None:
        """Close the window, if it is open."""
        if self.is_open:
            self._pygame.display.quit()
            self.is_open = False
