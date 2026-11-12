"""Verify root/headless SAPIEN physics and actual camera pixels without task assets."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import sapien
import sapien.render

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / '.runtime/validation'
    output.mkdir(parents=True, exist_ok=True)
    scene = sapien.Scene()
    scene.set_timestep(1 / 240)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [1, 1, 1])
    material = sapien.render.RenderMaterial()
    material.base_color = [0.8, 0.1, 0.1, 1]
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.15, 0.15, 0.15])
    builder.add_box_visual(half_size=[0.15, 0.15, 0.15], material=material)
    cube = builder.build(name='cube')
    cube.set_pose(sapien.Pose([0, 0, 0.3]))
    scene.add_ground(0)
    camera = scene.add_camera('headless', 128, 128, 1.0, 0.01, 10)
    camera.set_pose(sapien.Pose([-1.5, 0, 0.3]))
    start_z = float(cube.get_pose().p[2])
    for _ in range(10):
        scene.step()
    scene.update_render()
    camera.take_picture()
    pixels = camera.get_picture('Color')[..., :3]
    if not np.isfinite(pixels).all() or float(pixels.std()) < 0.01:
        raise RuntimeError('Camera image invalid or uniform')
    if not float(cube.get_pose().p[2]) < start_z:
        raise RuntimeError('Physics gravity check failed')
    Image.fromarray((np.clip(pixels, 0, 1) * 255).astype('uint8')).save(output / 'headless-render.png')
    result = dict(status='passed', uid=os.geteuid(), physics=True,
                  camera_shape=list(pixels.shape), pixel_std=float(pixels.std()))
    (output / 'headless-render.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
