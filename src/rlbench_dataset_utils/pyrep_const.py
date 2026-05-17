from enum import Enum

from . import coppeliasim_const as sim


class RenderMode(Enum):
    OPENGL = sim.sim_rendermode_opengl
    OPENGL_AUXILIARY = sim.sim_rendermode_auxchannels
    OPENGL_COLOR_CODED = sim.sim_rendermode_colorcoded
    POV_RAY = sim.sim_rendermode_povray
    EXTERNAL = sim.sim_rendermode_extrenderer
    EXTERNAL_WINDOWED = sim.sim_rendermode_extrendererwindowed
    OPENGL3 = sim.sim_rendermode_opengl3
    OPENGL3_WINDOWED = sim.sim_rendermode_opengl3windowed