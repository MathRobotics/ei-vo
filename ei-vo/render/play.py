from .render_mj import play as render_play

def play(model_path: str, traj, slow=1.0, hz=240.0, loop=False):
    render_play(model_path, traj, slow, hz, loop)