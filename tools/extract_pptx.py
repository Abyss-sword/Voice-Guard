import sys
from pptx import Presentation

path = sys.argv[1]
p = Presentation(path)
print("slide size:", p.slide_width, p.slide_height, "EMU")
print("n_slides:", len(p.slides))
for i, s in enumerate(p.slides, 1):
    print("\n" + "=" * 70)
    print(f"[SLIDE {i}]  layout={s.slide_layout.name}")
    shapes = sorted(s.shapes, key=lambda sh: (sh.top or 0, sh.left or 0))
    for sh in shapes:
        kind = sh.shape_type
        txt = ""
        if sh.has_text_frame:
            txt = "\n".join(par.text for par in sh.text_frame.paragraphs)
        if sh.has_table:
            rows = []
            for r in sh.table.rows:
                rows.append(" | ".join(c.text.replace("\n", " / ") for c in r.cells))
            txt = "\n".join(rows)
        if txt.strip():
            print(f"  <{kind}> {txt}")
    if s.has_notes_slide and s.notes_slide.notes_text_frame.text.strip():
        print("  [NOTES]", s.notes_slide.notes_text_frame.text)
