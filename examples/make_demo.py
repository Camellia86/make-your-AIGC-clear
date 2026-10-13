"""Create a fully synthetic illustration for the mask/projection workflow.

The two images are drawn with Pillow.  They are demonstration inputs, not
outputs from an image model.  No photographs, model weights, or network access
are needed.  Run from the repository root, for example:

    python examples/make_demo.py --output-dir demo
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


SIZE = (512, 512)
CLOTHING = [(88, 290), (190, 252), (322, 252), (424, 290), (464, 496), (48, 496)]
HAIR = [(146, 82), (176, 50), (226, 34), (290, 34), (338, 55), (370, 90),
        (384, 184), (370, 298), (328, 330), (312, 215), (198, 215),
        (180, 328), (140, 290), (128, 174)]


def draw_scene(extra_strokes):
    """Return a geometric portrait with controlled hair and clothing detail."""
    image = Image.new("RGB", SIZE, (232, 229, 218))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 354, 511, 511), fill=(217, 215, 204))
    draw.ellipse((52, 58, 460, 466), fill=(225, 222, 209))

    draw.polygon(CLOTHING, fill=(66, 86, 105))
    draw.polygon([(88, 290), (190, 252), (230, 302), (167, 496), (48, 496)],
                 fill=(57, 75, 92))
    draw.polygon([(322, 252), (424, 290), (464, 496), (342, 496), (282, 302)],
                 fill=(75, 96, 112))
    for x in range(100, 421, 36):
        draw.line((x, 326, x - 24, 485), fill=(86, 106, 123), width=2)
    draw.line((256, 302, 256, 495), fill=(43, 62, 78), width=4)
    for y in (338, 378, 418, 458):
        draw.ellipse((250, y, 262, y + 12), fill=(158, 171, 174))

    # Redundant short highlights are the only clothing detail being removed.
    if extra_strokes:
        for row in range(5):
            for column in range(9):
                x = 100 + column * 34 + (row % 2) * 9
                y = 327 + row * 32
                draw.line((x, y, x + 12, y - 4), fill=(174, 185, 186), width=2)
                draw.line((x + 2, y + 5, x + 9, y + 3), fill=(133, 153, 166), width=1)

    draw.polygon(HAIR, fill=(56, 48, 53))
    draw.polygon([(146, 82), (176, 50), (226, 34), (237, 91), (175, 159),
                  (180, 328), (140, 290), (128, 174)], fill=(67, 55, 58))
    for line in [((164, 91), (149, 257)), ((176, 106), (167, 295)),
                 ((344, 87), (365, 262)), ((332, 110), (348, 294))]:
        draw.line(line, fill=(111, 89, 84), width=3)
    if extra_strokes:
        for i in range(12):
            y = 101 + i * 15
            draw.line((145 + i % 3 * 7, y, 157 + i % 3 * 7, y - 6),
                      fill=(171, 139, 119), width=2)
            draw.line((348 + i % 3 * 6, y, 359 + i % 3 * 6, y + 7),
                      fill=(171, 139, 119), width=2)

    # Face, neck, and hands are drawn identically in both images.
    draw.rounded_rectangle((220, 211, 292, 292), radius=22, fill=(198, 154, 124))
    draw.ellipse((182, 88, 332, 248), fill=(221, 181, 147))
    draw.polygon([(183, 139), (190, 97), (224, 76), (296, 77), (329, 106),
                  (330, 134), (292, 105), (249, 98), (219, 122)], fill=(56, 48, 53))
    draw.arc((202, 143, 233, 162), 195, 345, fill=(72, 60, 57), width=3)
    draw.arc((278, 143, 309, 162), 195, 345, fill=(72, 60, 57), width=3)
    draw.line((256, 160, 251, 184, 261, 185), fill=(180, 135, 111), width=2)
    draw.arc((233, 185, 280, 214), 20, 155, fill=(130, 81, 77), width=3)
    draw.polygon([(190, 252), (220, 268), (256, 302), (226, 322), (177, 275)],
                 fill=(205, 215, 212))
    draw.polygon([(322, 252), (292, 268), (256, 302), (286, 322), (335, 275)],
                 fill=(223, 229, 222))
    draw.rounded_rectangle((72, 396, 115, 472), radius=18, fill=(211, 169, 135))
    draw.rounded_rectangle((397, 396, 440, 472), radius=18, fill=(211, 169, 135))
    for x in (84, 94, 104, 409, 419, 429):
        draw.line((x, 429, x, 459), fill=(181, 134, 110), width=1)
    return image


def make_demo(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    paths = [output_dir / name for name in ("source.png", "proposal.png", "scope.json")]
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite: {}".format(existing[0]))
    output_dir.mkdir(parents=True, exist_ok=True)

    source = draw_scene(extra_strokes=True)
    proposal = draw_scene(extra_strokes=False)
    # This deliberate background change demonstrates restoration by projection.
    ImageDraw.Draw(proposal).rectangle((24, 24, 51, 51), fill=(180, 198, 204))
    scope = {
        "coordinate_size": list(SIZE),
        "allow": [
            {"name": "clothing", "points": [list(point) for point in CLOTHING]},
            {"name": "hair", "points": [list(point) for point in HAIR]},
        ],
        "protect": [
            {"name": "face_and_neck", "points": [[178, 84], [336, 84], [336, 294], [178, 294]]},
            {"name": "left_hand", "points": [[68, 390], [120, 390], [120, 478], [68, 478]]},
            {"name": "right_hand", "points": [[392, 390], [445, 390], [445, 478], [392, 478]]},
        ],
        "feather_px": 7,
    }
    source.save(paths[0])
    proposal.save(paths[1])
    paths[2].write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        paths = make_demo(args.output_dir, args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, "error: {}\n".format(exc))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
