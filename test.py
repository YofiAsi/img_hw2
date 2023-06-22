from pathlib import Path
import subprocess


scenes = Path("./scenes")
for scne in scenes.rglob("*.txt"):
    pic = f"output\\result_{scne.with_suffix('.png').name}"
    args = ["python", "ray_tracer.py", str(scne), pic]

    try:
        subprocess.run(args)
    except Exception as e:
        print(f"{scenes.name} - Error: {e}")
        continue