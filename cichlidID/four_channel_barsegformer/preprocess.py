from pathlib import Path
from PIL import Image
import shutil

def preprocess_dataset(data_dir, output_dir, image_size=224):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    seg_dir = data_dir / "segmentations"
    out_seg_dir = output_dir / "segmentations"

    count = 0
    for type_dir in ["F3", "F4-Cross"]:
        type_path = data_dir / type_dir
        if not type_path.exists():
            continue
        for tank_dir in sorted(type_path.iterdir()):
            if not tank_dir.is_dir():
                continue
            for fish_dir in sorted(tank_dir.iterdir()):
                if not fish_dir.is_dir():
                    continue

                # Output dirs
                out_fish = output_dir / type_dir / tank_dir.name / fish_dir.name
                out_fish.mkdir(parents=True, exist_ok=True)
                out_fish_seg = out_seg_dir / type_dir / tank_dir.name / fish_dir.name
                out_fish_seg.mkdir(parents=True, exist_ok=True)

                for jpg in sorted(fish_dir.glob("*.JPG")):
                    # Resize and save JPG
                    img = Image.open(jpg).convert("RGB")
                    img = img.resize((image_size, image_size), Image.LANCZOS)
                    img.save(out_fish / jpg.name, quality=95)

                    # Resize and save matching mask
                    png = seg_dir / type_dir / tank_dir.name / fish_dir.name / (jpg.stem + "_seg.png")
                    if png.exists():
                        mask = Image.open(png).convert("L")
                        mask = mask.resize((image_size, image_size), Image.NEAREST)
                        mask.save(out_fish_seg / png.name)

                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} images...")

    print(f"Done. Total: {count} images saved to {output_dir}")

preprocess_dataset("data_raw_new", "data_224")