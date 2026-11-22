import types

from PIL import Image

from localmind.rendering import render_pages
from tests.pdf_factory import make_pdf


def test_render_pdf_page(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(make_pdf(["Hello world page", ""]))
    [(no, img)] = list(render_pages(f, [2], dpi=72))
    assert no == 2 and img.mode == "RGB" and img.size == (612, 792)


def test_render_image_frame(tmp_path):
    f = tmp_path / "fax.tif"
    frames = [Image.new("L", (40, 20), c) for c in (0, 128, 255)]
    frames[0].save(f, save_all=True, append_images=frames[1:])
    [(no, img)] = list(render_pages(f, [2]))
    assert no == 2 and img.mode == "RGB" and img.getpixel((0, 0)) == (128, 128, 128)


def test_exif_rotation_is_applied(tmp_path):
    f = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # taken sideways
    Image.new("RGB", (60, 20), "white").save(f, exif=exif)
    [(_, img)] = list(render_pages(f, [1]))
    assert img.size == (20, 60)


def test_render_is_lazy(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(make_pdf(["", "", ""]))
    gen = render_pages(f, [1, 2, 3], dpi=72)
    assert isinstance(gen, types.GeneratorType)
    assert next(gen)[0] == 1
