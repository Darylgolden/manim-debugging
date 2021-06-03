from pathlib import Path
import moderngl
import numpy as np
import moderngl_window

MODULE_PATH = Path(__file__).parent.resolve()
SHADER_PATH = MODULE_PATH / "shaders"


class Test(moderngl_window.WindowConfig):
    title = "Quadratic Bézier Fill Test"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vertices = np.array([([ 1.,  1.,  0.], [ 1., -1.,  0.], [-1.,  1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([ 0.,  1.,  0.], [ 1.,  0.,  0.], [-1.,  0.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([-1.,  1.,  0.], [ 1.,  1.,  0.], [-1., -1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([-1.,  1.,  0.], [ 1.,  1.,  0.], [-1., -1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([-1.,  0.,  0.], [ 0.,  1.,  0.], [ 0., -1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([-1., -1.,  0.], [-1.,  1.,  0.], [ 1., -1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([-1., -1.,  0.], [-1.,  1.,  0.], [ 1., -1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([ 0., -1.,  0.], [-1.,  0.,  0.], [ 1.,  0.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([ 1., -1.,  0.], [-1., -1.,  0.], [ 1.,  1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([ 1., -1.,  0.], [-1., -1.,  0.], [ 1.,  1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([ 1.,  0.,  0.], [ 0., -1.,  0.], [ 0.,  1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.]),
       ([ 1.,  1.,  0.], [ 1., -1.,  0.], [-1.,  1.,  0.], [0., 0., 1.], [4.], [1., 1., 1., 1.])],
      dtype=[('point', '<f4', (3,)), ('prev_point', '<f4', (3,)), ('next_point', '<f4', (3,)), ('unit_normal', '<f4', (3,)), ('stroke_width', '<f4', (1,)), ('color', '<f4', (4,))])
        self.indices = None

        self.vbo = self.ctx.buffer(self.vertices.tobytes())
        # self.ibo = self.ctx.buffer(self.indices.tobytes())

        self.program = self.ctx.program(
            vertex_shader=open(SHADER_PATH / "quadratic_bezier_stroke/vert.glsl").read(),
            fragment_shader=open(SHADER_PATH / "quadratic_bezier_stroke/frag.glsl").read(),
            geometry_shader=open(SHADER_PATH / "quadratic_bezier_stroke/geom.glsl").read(),
        )

        self.vao = self.ctx.vertex_array(
            self.program,
            self.vbo,
            *self.vertices.dtype.names, # ('point', 'unit_normal', 'color', 'vert_index')
            # index_buffer=self.ibo,
        )

        print("Program Members:")
        for name, obj in self.program._members.items():
            print(f" * {name} {obj}")

        self.program["anti_alias_width"] = 0.011111111111111112
        self.program["camera_center"] = (0.0, 0.0, 0.0)
        self.program["camera_rotation"] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.program["focal_distance"] = 16.0
        self.program["frame_shape"] = (14.222222222222221, 8.0)
        self.program["gloss"] = 0.0
        self.program["is_fixed_in_frame"] = 0.0
        self.program["light_source_position"] = (-10.0, 10.0, 10.0)
        self.program["shadow"] = 0.0

    def render(self, time, frame_time):
        self.ctx.clear()
        self.vao.render(moderngl.TRIANGLES)


if __name__ == "__main__":
    Test.run()