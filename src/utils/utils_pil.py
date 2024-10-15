from PIL import Image


def get_concat_v(im1, im2, spacing: int = 0):
    dst = Image.new("RGB", (im1.width, im1.height + im2.height + spacing))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height + spacing))
    return dst


def get_concat_h(im1, im2, spacing: int = 0):
    dst = Image.new("RGB", (im1.width + im2.width + spacing, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + spacing, 0))
    return dst
