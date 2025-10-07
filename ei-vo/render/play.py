from typing import Optional, Tuple

from .render_mj import play as render_play


def play(model_path: str, traj, slow=1.0, hz=240.0, loop=False,
         record_path: Optional[str] = None, record_fps: Optional[float] = None,
         record_size: Optional[Tuple[int, int]] = None):
    render_play(
        model_path,
        traj,
        slow,
        hz,
        loop=loop,
        record_path=record_path,
        record_fps=record_fps,
        record_size=record_size,
    )
