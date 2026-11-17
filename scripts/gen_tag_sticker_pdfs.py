"""
Print-ready vector PDFs of ArUco tags for sticker printing.

Each sticker is one page of exactly sticker_mm x sticker_mm: the tag's
black square (tag_mm, the size in aruco_config.yaml) centered on a white
quiet zone, with a small ID caption in the margin. Also writes one PDF
per tag, A4 sheets with the stickers tiled and cut lines (for a print
shop), and an A4 size-check page (print it at 100% and measure).

python scripts/gen_tag_sticker_pdfs.py -o ~/Desktop/aruco_print/stickers_env \
    --ids 14-17,20-23,26-29,32-35,38-41,44-47 --ids 30:RIGHT\ PARK,31:LEFT\ PARK
"""
# %%
import os
import pathlib
import click
import cv2
import numpy as np

MM = 72.0 / 25.4  # PDF points per millimeter


# %%
def marker_bits(dictionary, tag_id: int) -> np.ndarray:
    """
    (n+2, n+2) bool, True = white, including the black border, row 0 = top.
    """
    n = dictionary.markerSize + 2
    return cv2.aruco.generateImageMarker(dictionary, tag_id, n) > 127


def sticker_content(bits: np.ndarray, sticker_mm: float, tag_mm: float, caption: str,
        offset_mm=(0.0, 0.0)) -> bytes:
    """
    PDF content stream of one sticker, its bottom left corner at offset_mm.
    """
    n = bits.shape[0]
    cell = tag_mm / n * MM
    x0 = y0 = (sticker_mm - tag_mm) / 2 * MM
    ops = ['q', f'1 0 0 1 {offset_mm[0] * MM:.3f} {offset_mm[1] * MM:.3f} cm',
           '1 g', f'0 0 {sticker_mm * MM:.3f} {sticker_mm * MM:.3f} re f',
           '0 g', f'{x0:.3f} {y0:.3f} {tag_mm * MM:.3f} {tag_mm * MM:.3f} re f', '1 g']
    # all white cells as one path filled once (nonzero union): separately filled
    # abutting cells show anti-aliasing hairlines between them in viewers and RIPs
    cells = [f'{x0 + c * cell:.3f} {y0 + (n - 1 - r) * cell:.3f} {cell:.3f} {cell:.3f} re'
        for r in range(n) for c in range(n) if bits[r, c]]
    if cells:
        ops += cells + ['f']
    # caption: small gray text in the bottom margin, well away from the black square
    caption_y = min(3.5, (sticker_mm - tag_mm) / 2 * 0.3)
    ops += ['0.55 g', 'BT', '/F1 6.5 Tf', f'{x0:.3f} {caption_y * MM:.3f} Td', f'({caption}) Tj', 'ET', 'Q']
    return '\n'.join(ops).encode('latin-1')


A4 = (210.0, 297.0)


def a4_sheets(stickers, sticker_mm: float, tag_mm: float, margin_mm: float = 5.0, footer_mm: float = 14.0):
    """
    A4 pages with stickers tiled edge to edge, dashed cut lines on the
    sticker edges and a footer with print instructions and a 50 mm ruler.
    stickers: list of (bits, caption)
    """
    cols = int((A4[0] - 2 * margin_mm) // sticker_mm)
    rows = int((A4[1] - 2 * margin_mm - footer_mm) // sticker_mm)
    assert cols >= 1 and rows >= 1, "sticker larger than an A4 page"
    per_page = cols * rows
    x_left = (A4[0] - cols * sticker_mm) / 2
    y_bottom = footer_mm + margin_mm + (A4[1] - 2 * margin_mm - footer_mm - rows * sticker_mm) / 2
    n_pages = -(-len(stickers) // per_page)
    pages = list()
    for p in range(n_pages):
        chunk = stickers[p * per_page:(p + 1) * per_page]
        ops = list()
        for k, (bits, caption) in enumerate(chunk):
            r, c = divmod(k, cols)
            ops.append(sticker_content(bits, sticker_mm, tag_mm, caption,
                (x_left + c * sticker_mm, y_bottom + (rows - 1 - r) * sticker_mm)).decode('latin-1'))
        # cut lines on all sticker edges of the used cells, extended to the page edge
        used_rows = -(-len(chunk) // cols)
        ops += ['0.6 G', '0.3 w', '[2 2] 0 d']
        xs = [x_left + c * sticker_mm for c in range(cols + 1)]
        ys = [y_bottom + (rows - r) * sticker_mm for r in range(used_rows + 1)]
        for x in xs:
            ops.append(f'{x * MM:.3f} {(ys[-1] - margin_mm) * MM:.3f} m {x * MM:.3f} {A4[1] * MM:.3f} l S')
        for y in ys:
            ops.append(f'0 {y * MM:.3f} m {A4[0] * MM:.3f} {y * MM:.3f} l S')
        ops.append('[] 0 d')
        # footer: instructions and a 50 mm ruler
        ops += ['0 g', 'BT', '/F1 8 Tf', f'{12 * MM:.3f} {(margin_mm + 6) * MM:.3f} Td',
            f'(PRINT AT 100% / ACTUAL SIZE - no fit to page. Black squares = {tag_mm:g} mm, '
            f'stickers {sticker_mm:g} x {sticker_mm:g} mm. Cut on dashed lines. Matte. '
            f'Sheet {p + 1}/{n_pages}) Tj', 'ET', '0 G', '0.4 w']
        rx, ry = 12 * MM, (margin_mm + 1) * MM
        ops.append(f'{rx:.3f} {ry:.3f} m {rx + 50 * MM:.3f} {ry:.3f} l S')
        for i in range(0, 51):
            h = (3 if i % 10 == 0 else 1.8 if i % 5 == 0 else 1) * MM
            ops.append(f'{rx + i * MM:.3f} {ry:.3f} m {rx + i * MM:.3f} {ry + h:.3f} l S')
        ops += ['BT', '/F1 6 Tf', f'{rx + 52 * MM:.3f} {ry:.3f} Td', '(50 mm) Tj', 'ET']
        pages.append((A4[0] * MM, A4[1] * MM, '\n'.join(ops).encode('latin-1')))
    return pages, per_page


def size_check_content(tag_mm: float) -> bytes:
    """
    A4 page with a 100 mm ruler and a tag_mm reference square.
    """
    ops = ['0 g', '0.5 w']
    x, y = 30 * MM, 200 * MM
    ops.append(f'{x:.3f} {y:.3f} m {x + 100 * MM:.3f} {y:.3f} l S')
    for i in range(101):
        h = (5 if i % 10 == 0 else 3 if i % 5 == 0 else 1.5) * MM
        ops.append(f'{x + i * MM:.3f} {y:.3f} m {x + i * MM:.3f} {y + h:.3f} l S')
    ops += ['BT', '/F1 9 Tf', f'{x:.3f} {y + 9 * MM:.3f} Td', '(0) Tj', f'{98 * MM:.3f} 0 Td', '(100 mm) Tj', 'ET']
    s = tag_mm * MM
    ops += [f'{x:.3f} {110 * MM:.3f} {s:.3f} {s:.3f} re f']
    lines = [
        'SIZE CHECK - print at 100% / actual size, no "fit to page".',
        f'The black square must measure {tag_mm:.1f} mm and the ruler 100 mm (calipers).',
        f'Every tag on the sticker pages: black square = {tag_mm:.1f} mm, outer edge of the black.',
        'Matte finish (no gloss), flat rigid surface, no lamination wrinkles.',
    ]
    ops += ['BT', '/F1 10 Tf', f'{x:.3f} {260 * MM:.3f} Td', '14 TL']
    ops += [f'({t}) Tj T*' for t in lines]
    ops.append('ET')
    return '\n'.join(ops).encode('latin-1')


def write_pdf(path: pathlib.Path, pages):
    """
    Minimal PDF 1.4 writer. pages: list of (width_pt, height_pt, content bytes).
    """
    objects = list()
    add = lambda data: objects.append(data) or len(objects)
    catalog = add(None)
    pages_id = add(None)
    font = add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>')
    page_ids = list()
    for w, h, content in pages:
        stream = add(b'<< /Length %d >>\nstream\n' % len(content) + content + b'\nendstream')
        page_ids.append(add(f'<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {w:.3f} {h:.3f}] '
            f'/Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R >>'.encode()))
    objects[catalog - 1] = f'<< /Type /Catalog /Pages {pages_id} 0 R >>'.encode()
    objects[pages_id - 1] = (f'<< /Type /Pages /Count {len(page_ids)} /Kids ['
        + ' '.join(f'{i} 0 R' for i in page_ids) + '] >>').encode()
    out = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = list()
    for i, data in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b'%d 0 obj\n' % i + data + b'\nendobj\n'
    xref = len(out)
    out += b'xref\n0 %d\n0000000000 65535 f \n' % (len(objects) + 1)
    out += b''.join(b'%010d 00000 n \n' % o for o in offsets)
    out += b'trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (len(objects) + 1, catalog, xref)
    path.write_bytes(bytes(out))


def parse_ids(specs):
    """
    '14-17,20' and '30:RIGHT PARK' -> [(id, role)]
    """
    out = list()
    for spec in specs:
        for part in spec.split(','):
            part, _, role = part.partition(':')
            if '-' in part:
                a, b = part.split('-')
                out += [(i, role or 'ENV') for i in range(int(a), int(b) + 1)]
            else:
                out.append((int(part), role or 'ENV'))
    return out


# %%
@click.command()
@click.option('-o', '--output', required=True, help='Output directory')
@click.option('--ids', multiple=True, required=True, help='e.g. 14-17,20 or 30:RIGHT PARK')
@click.option('-t', '--tag_mm', type=float, default=80.0, help='Black square edge, as in aruco_config.yaml')
@click.option('-s', '--sticker_mm', type=float, default=110.0, help='Sticker edge (tag + white quiet zone)')
@click.option('-n', '--name', default='env_tags', help='Combined PDF name prefix')
@click.option('-a', '--a4_sticker_mm', type=float, multiple=True, default=[100.0, 110.0],
    help='Sticker edge for the A4 sheet layouts, one PDF each')
def main(output, ids, tag_mm, sticker_mm, name, a4_sticker_mm):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tags = parse_ids(ids)
    assert len({i for i, _ in tags}) == len(tags), "duplicate tag ids"
    assert all(0 <= i < 50 for i, _ in tags), "DICT_4X4_50 ids are 0..49"
    module_mm = tag_mm / (dictionary.markerSize + 2)
    for s in [sticker_mm, *a4_sticker_mm]:
        quiet = (s - tag_mm) / 2
        assert quiet >= 0.5 * module_mm, f"{s:g} mm sticker: quiet zone {quiet:g} mm below half a module"
        if quiet < module_mm:
            print(f"note: {s:g} mm sticker has a {quiet:g} mm white border, "
                f"{quiet / module_mm:.2f} of a module (1 module = {module_mm:.1f} mm recommended)")

    out_dir = pathlib.Path(os.path.expanduser(output))
    out_dir.joinpath('individual').mkdir(parents=True, exist_ok=True)
    size = sticker_mm * MM
    pages = list()
    for tag_id, role in tags:
        caption = f'ID {tag_id}  |  {tag_mm:g} mm  |  DICT_4X4_50  |  {role}'
        page = (size, size, sticker_content(marker_bits(dictionary, tag_id), sticker_mm, tag_mm, caption))
        pages.append(page)
        write_pdf(out_dir.joinpath('individual', f'tag_{tag_id:02d}_{tag_mm:g}mm.pdf'), [page])
    for s in a4_sticker_mm:
        stickers = [(marker_bits(dictionary, i), f'ID {i}  |  {tag_mm:g} mm  |  DICT_4X4_50  |  {role}')
            for i, role in tags]
        sheets, per_page = a4_sheets(stickers, s, tag_mm)
        path = out_dir.joinpath(f'{name}_A4_{per_page}up_{s:g}mm.pdf')
        write_pdf(path, sheets)
        print(f"A4: {per_page} stickers per sheet ({s:g} mm), {len(sheets)} sheets -> {path}")
    combined = out_dir.joinpath(f'{name}_{len(tags)}x_{sticker_mm:g}mm_stickers.pdf')
    write_pdf(combined, pages)
    write_pdf(out_dir.joinpath('size_check_A4.pdf'), [(210 * MM, 297 * MM, size_check_content(tag_mm))])
    print(f"{len(tags)} stickers ({sticker_mm:g} x {sticker_mm:g} mm, tag {tag_mm:g} mm, "
        f"quiet zone {(sticker_mm - tag_mm) / 2:g} mm) -> {combined}")

# %%
if __name__ == "__main__":
    main()
