"""Lossless RGB/depth samples plus timestamped videos from every scene camera."""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def encode(folder, output, stamps, slow_gif=True):
    if not stamps:
        raise RuntimeError('No frames for ' + str(output))
    durations = [max(.01, b - a) for a, b in zip(stamps, stamps[1:])]
    durations.append(durations[-1] if durations else .1)
    lines = []
    for index, duration in enumerate(durations):
        lines.extend([f"file '{index:06d}.png'", f'duration {duration:.9f}'])
    lines.append(f"file '{len(stamps)-1:06d}.png'")
    manifest = folder / 'frames.ffconcat'
    manifest.write_text('\n'.join(lines) + '\n')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
                    '-safe', '0', '-i', str(manifest.resolve()), '-vf', 'fps=30',
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(output.with_suffix('.mp4').resolve())], check=True)
    if slow_gif:
        images = []
        for index in range(len(stamps)):
            with Image.open(folder / f'{index:06d}.png') as frame:
                image = frame.convert('RGB')
                if image.width > 720:
                    image = image.resize((720, round(image.height * 720 / image.width)))
                images.append(image)
        delay = [max(20, round(3000 * duration / 10) * 10) for duration in durations]
        delay[-1] += 1200
        images[0].save(output.with_suffix('.gif'), save_all=True, append_images=images[1:],
                       duration=delay, loop=0, disposal=2, optimize=False)
    return dict(frames=len(stamps), start_time=stamps[0], end_time=stamps[-1],
                mp4=output.with_suffix('.mp4').name,
                gif=output.with_suffix('.gif').name if slow_gif else None)


class CameraRecording:
    def __init__(self, output, enabled=True):
        self.output = Path(output)
        self.stamps = []
        self.enabled = enabled

    def observe(self, observation):
        if not self.enabled:
            return
        stamp = float(observation['image_time'])
        if self.stamps and abs(stamp - self.stamps[-1]) < 1e-8:
            return
        index = len(self.stamps)
        for name in ('front', 'side'):
            folder = self.output / 'camera_frames' / name
            folder.mkdir(parents=True, exist_ok=True)
            image = Image.fromarray(observation[name])
            image.save(folder / f'{index:06d}.png')
            if index == 0:
                image.save(self.output / f'initial_{name}.png')
            image.save(self.output / f'final_{name}.png')
            depth = np.asarray(observation[name + '_depth'], np.float32)
            folder = self.output / 'camera_frames' / (name + '_depth')
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f'{index:06d}.npy', depth)
            # Fixed 0.1--1.2 metre grayscale range across all tasks; white is near.
            valid = np.isfinite(depth) & (depth > 0)
            gray = np.zeros(depth.shape, np.uint8)
            gray[valid] = np.clip((1.2 - depth[valid]) / 1.1 * 255, 0, 255).astype(np.uint8)
            Image.fromarray(gray).convert('RGB').save(folder / f'{index:06d}.png')
        self.stamps.append(stamp)

    def save_observer(self, name, source_frames):
        frames = [(stamp, image) for stamp, image in source_frames
                  if self.stamps[0] - .11 <= stamp <= self.stamps[-1] + .11]
        folder = self.output / 'camera_frames' / name
        folder.mkdir(parents=True, exist_ok=True)
        for index, (_, rgb) in enumerate(frames):
            image = Image.fromarray(rgb)
            image.save(folder / f'{index:06d}.png')
            if index == 0:
                image.save(self.output / f'initial_{name}.png')
            image.save(self.output / f'final_{name}.png')
        return encode(folder, self.output / ('policy_' + name), [v[0] for v in frames])

    def finish(self, display_frames, opposite_frames=None):
        files = {}
        for name in ('front', 'side', 'front_depth', 'side_depth'):
            files[name] = encode(self.output / 'camera_frames' / name,
                                 self.output / ('policy_' + name), self.stamps,
                                 slow_gif=not name.endswith('_depth'))
        files['overview'] = self.save_observer('overview', display_frames)
        if opposite_frames is not None:
            files['opposite'] = self.save_observer('opposite', opposite_frames)
        metadata = dict(files=files, image_times=self.stamps,
            rgb_topics=['/camera/front/image', '/camera/side/image', '/bimanual/display/image'] +
                       (['/camera/opposite/image'] if opposite_frames is not None else []),
            recording_only_views=['overview'] + (['opposite'] if opposite_frames is not None else []),
            policy_rgb_views=['front', 'side', 'previous_front'],
            depth_topics=['/camera/front/depth', '/camera/side/depth'],
            depth_display='Fixed grayscale: 0.1 m white to 1.2 m black; invalid black. Raw float32 metres saved as NPY.',
            mp4_timing='simulation time; sampled frames presented at 30 fps, not 30 fps camera acquisition',
            gif_timing='one-third simulation speed plus 1.2 second final pause')
        (self.output / 'camera_recording.json').write_text(json.dumps(metadata, indent=2))
        return metadata
