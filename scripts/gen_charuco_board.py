# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
from PIL import Image
from umi.common.cv_util import get_charuco_board, draw_charuco_board

# %%
@click.command()
@click.option('-o', '--output', required=True, help='Output pdf path')
@click.option('-to', '--tag_id_offset', type=int, default=50)
def main(output, tag_id_offset):
    dpi = 300
    board = get_charuco_board(tag_id_offset=tag_id_offset)
    board_img = draw_charuco_board(board=board, dpi=dpi)
    im = Image.fromarray(board_img)
    im.save(output, resolution=dpi)

# %%
if __name__ == '__main__':
    main()
