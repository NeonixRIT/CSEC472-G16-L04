from PIL import Image, ImageFilter

import os
from pathlib import Path

from time import perf_counter


def batch(iterable, n):
    """
    batch([1,2,3,4,5,6,7], 3) => [(1,2,3), (4,5,6), (7,)]
    """
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


def get_orientation(image: Image.Image) -> str:
    """
    each image has 32 white pixels at the bottom of the image
    this function checks where these pixels are located to determine the orientation of the image then
    returns the angle of rotation needed to make the image upright.
    """
    raw = image.tobytes()
    row_chunks = list(batch(raw, 512))
    col_chunks = [bytes([row[i] for row in row_chunks]) for i in range(512)]
    row_nums = [row_chunk.count(b'\xff') for row_chunk in row_chunks]
    col_nums = [col_chunk.count(b'\xff') for col_chunk in col_chunks]
    is_top = int(all([row_num == 512 for row_num in row_nums[:32]]) and all([col_num >= 32 for col_num in col_nums[:32]]))
    is_bottom = int(all([row_num == 512 for row_num in row_nums[-32:]]) and all([col_num >= 32 for col_num in col_nums[-32:]]))
    is_left = int(all([col_num == 512 for col_num in col_nums[:32]]) and all([row_num >= 32 for row_num in row_nums[:32]]))
    is_right = int(all([col_num == 512 for col_num in col_nums[-32:]]) and all([row_num >= 32 for row_num in row_nums[-32:]]))
    if is_top:
        return 180
    if is_bottom:
        return 0
    if is_left:
        return 90
    if is_right:
        return 270


def get_samples_data() -> dict:
    """
    load all fingerprint images and their corresponding data into memory.
    """
    image_data: dict[int, dict[str, Image.Image | str | int]] = {i: {'f': [], 's': [], 'f_path': '', 's_path': '', 'f_txt_path': '', 's_txt_path': '', 'gender': '', 'class': '', 'history': '', 'finger': 0} for i in range(1, 2001)}
    figs_path = Path(__file__).parent / 'NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt'
    for root, _, files in os.walk(figs_path):
        for file in files:
            if file.endswith('.png'):
                image_type = file[0] + '_path'
                image_id = int(file[1:].split('.')[0].split('_')[0])
                image_data[image_id][image_type] = str(Path(root, file))
                image_data[image_id][image_type[0]] = Image.open(str(Path(root, file)))
                image_data[image_id]['finger'] = int(file.split('_')[1].split('.')[0])
            elif file.endswith('.txt'):
                image_id = int(file[1:].split('.')[0].split('_')[0])
                image_data[image_id][file[0] + '_txt_path'] = str(Path(root, file))
                with open(str(Path(root, file)), 'r') as f:
                    data = {k: v for k, v in [line.strip().split(': ') for line in f.readlines()]}
                    for k, v in data.items():
                        image_data[image_id][k.lower()] = v
    return image_data


def mask_image_thresh(image: Image.Image, thresh: int) -> Image.Image:
    """
    if pixel is above threshold, set to white, else set to black.
    """
    return image.convert('L').point(lambda x: 255 if x > thresh else 0, mode='1')


def average_images(images: list[Image.Image]) -> Image.Image:
    """
    Take list of images and return an image that is the average of all images.
    """
    result_image = []
    for image in images:
        image = image.convert('1').tobytes()
        for i in range(len(image)):
            if i > len(result_image) - 1:
                result_image.append(0)
            result_image[i] += image[i]
    for i in range(len(result_image)):
        result_image[i] = result_image[i] // len(images)
    return Image.frombytes('1', (512, 512), bytes(result_image))


def thinner_lines(image: Image.Image, factor: int) -> Image.Image:
    """
    Turn image into binary string. 1 = white, 0 = black.
    take all string of consecutive 0s and replace `factor` with 1s on each side.
    This results in wider crevases in fingerprint lines.
    """
    threshold = factor * 2 + 1
    binary = bin(int(image.tobytes().hex(), 16))[2:]
    pad = '1' * factor
    result = ''
    i = 0
    for bit in binary:
        if bit == '0':
            j = i
            tmp = ''
            while j < len(binary) and binary[j] == '0':
                tmp += '0'
                j += 1
            if len(tmp) > threshold:
                result += pad + tmp[factor:-factor] + pad
            else:
                # result += tmp
                result += '1' * len(tmp)
            i += len(tmp)
        else:
            result += '1'
            i += 1
    # result = ''.join(['1' if bit == '0' else '0' for bit in result])
    return Image.frombytes('1', (512, 512), int(result, 2).to_bytes(32768, 'big'))


def denoise(image: Image.Image) -> Image.Image:
    """
    if only 25% of pixels in row are black, set all pixels in row to white.
    do the same for columns.
    """
    bin_rows = list(batch(bin(int.from_bytes(image.tobytes()))[2:].zfill(262144), 512))
    for i, row in enumerate(bin_rows):
        if row.count('0') == len(row):
            bin_rows[i] = '1' * len(row)
            continue
        if row.count('0') < 128:
            bin_rows[i] = '1' * len(row)
            continue
    for i in range(512):
        if sum([1 for row in bin_rows if row[i] == '0']) < 128:
            for j in range(len(bin_rows)):
                bin_rows[j] = bin_rows[j][:i] + '1' + bin_rows[j][i + 1 :]
    return Image.frombytes('1', (512, 512), int(''.join(bin_rows), 2).to_bytes(32768, 'big'))


def get_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    """
    get the bounds of the fingerprint in the image by looking for the second and second to last row and column
    whose compressed value is 0.
    """
    err = 0
    width, height = image.size
    bin_rows = list(batch(bin(int.from_bytes(image.tobytes()))[2:].zfill(width * height), width))
    compressed = [[] for _ in range(128)]
    # compress 4x4 chunks of image into 1 bit so total image size is 128x128
    for i in range(0, len(bin_rows), 4):
        for j in range(0, len(bin_rows[i]), 4):
            chunk = [bin_rows[k][j : j + 4] for k in range(i, i + 4)]
            chunk_avg = sum([sum([int(val) for val in row]) for row in chunk])
            compressed[i // 4].append('0' if chunk_avg < 4 else '1')
    # get the second and second to last row and column whose compressed value is 0
    left = sorted(set([compressed[i].index('0', compressed[i].index('0')) for i in range(len(compressed)) if '0' in compressed[i]]))
    left = left[1 if len(left) > 1 else 0] * 4
    right = sorted(set([-compressed[i][::-1].index('0', compressed[i][::-1].index('0')) % len(compressed[i]) for i in range(len(compressed)) if '0' in compressed[i]]))
    right = right[-2 if len(right) > 1 else -1] * 4
    lower = sorted(set([i for i in range(len(compressed)) if '0' in compressed[i]]))
    lower = lower[1 if len(lower) > 1 else 0] * 4
    upper = sorted(set([i for i in range(len(compressed)) if '0' in compressed[i]]))
    upper = upper[-2 if len(upper) > 1 else -1] * 4
    if right < left:
        right = width - 16
        err = 1
    if upper < lower:
        upper = height - 16
        err = 1
    return left, lower, right, upper, err


def normalize_image(image: Image.Image, threshhold_modifier: int = 0) -> Image.Image:
    """
    Correct image orientation
    Enhance and sharpen to make masking easier
    Mask image so that it is only black and white based on the average pixel value +/- threshhold_modifier
    Denoise image by making rows and columns that are mostly white all white
    Get the bounds of the fingerprint in the image
    Crop the image to the bounds
    Resize the image back to 512x512
    """
    image = image.rotate(get_orientation(image))
    image = image.filter(ImageFilter.EDGE_ENHANCE)
    image = image.filter(ImageFilter.SHARPEN)

    masked = None
    thresh_base = sum(list(image.tobytes())) // 262144
    if threshhold_modifier > 0:
        thresh_1 = (thresh_base - threshhold_modifier) % 256
        thresh_2 = (thresh_base + threshhold_modifier) % 256

        a = mask_image_thresh(image, thresh_1)
        b = mask_image_thresh(image, thresh_2)
        masked = average_images([a, b])
    else:
        masked = mask_image_thresh(image, thresh_base % 256)

    denoised = denoise(masked)
    left, lower, right, upper, err = get_bounds(denoised)
    cropped = denoised.crop((left, lower, right, upper))
    res = cropped.resize((512, 512))
    while err:
        d2 = denoise(res)
        left, lower, right, upper, err = get_bounds(d2)
        cropped = d2.crop((left, lower, right, upper))
        res = cropped.resize((512, 512))
    return res


def main():
    print('Getting samples data...')
    error = None
    start = perf_counter()
    for _, data in get_samples_data().items():
        if _ == 1:
            stop = perf_counter()
            print(f'done in {stop - start:.2f}s.\n')

        # data['f'].show()
        # data['s'].show()
        try:
            f_image = normalize_image(data['f'])
            s_image = normalize_image(data['s'])
        except (Exception, KeyboardInterrupt) as e:
            print(f'Error: {e}')
            print(_)
            data['f'].show()
            data['s'].show()
            error = e
    print(f'Normalized all images in {perf_counter() - start:.2f}s.')
    if error:
        raise error


if __name__ == '__main__':
    main()
