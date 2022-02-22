import os
import cv2
import random
import argparse
import numpy as np
import tempfile
from PIL import Image, ImageFont, ImageDraw

images_dir_path = './'
output_image_path = 'output.png'
G_WIDTH = 800
G_HEIGHT = 600
G_GAP = 0
G_OFFSET = 0
G_ITER = 5
G_BOLD = 10
G_STYLE = 'default'
G_TEXT = ''
G_TEXT_POS = (int(G_WIDTH * 0.8), int(G_HEIGHT * 0.8))
G_FONT_DIR = os.path.join(os.path.dirname(__file__), 'fonts')
G_FONT_PATH = os.path.join(os.path.dirname(__file__), './fonts/OPPOSans-B-2.ttf')


def convert_path(path: str) -> str:
    return path.replace(r'\/'.replace(os.sep, ''), os.sep)


def get_argument():
    parser = argparse.ArgumentParser(description='必须给定一个目录,里面包含至少一张图片')
    parser.add_argument('images_dir_path', type=str, help='图片目录')
    parser.add_argument("-s", "--size", type=str, help="输出图片的大小，默认为800x600", default='800x600')
    parser.add_argument("-o", "--output", type=str, help='输出图片路径', default='./output.png')
    parser.add_argument("-g", "--gap", type=int, help='图片之间的间隙', default=0)
    parser.add_argument("-f", "--offset", type=float, help='随机拼接图片时的图片位置偏移概率', default=0)
    parser.add_argument("-i", "--iter", type=int, help='迭代次数', default=5)
    parser.add_argument("-t", "--type", type=str, help='样式', default='default')
    parser.add_argument("-b", "--bold", type=int, help='边框大小', default=10)
    parser.add_argument("--text", type=str, help='要添加的字体', default='')
    parser.add_argument("--text_pos", type=str, help='要添加的字体显示的位置', default='')

    args = parser.parse_args()
    global images_dir_path, output_image_path, G_HEIGHT, \
        G_WIDTH, G_GAP, G_ITER, G_OFFSET, G_STYLE, G_BOLD, G_TEXT, \
        G_TEXT_POS
    images_dir_path = convert_path(os.path.join(os.path.abspath('./'), args.images_dir_path))
    output_image_path = convert_path(os.path.join(os.path.abspath('./'), args.output))
    try:
        G_WIDTH = int(args.size.split('x')[0])
        G_HEIGHT = int(args.size.split('x')[1])
        if args.type == 'cover_font':
            G_TEXT_POS = (int(G_WIDTH * 0.8), int(G_HEIGHT * 0.8))
            G_TEXT_POS = (
                int(float(args.text_pos.split('x')[0]) * G_WIDTH), int(float(args.text_pos.split('x')[1]) * G_HEIGHT))
    except:
        print('Check your input of size, is it like this "800x600"?\nif not,I will use 800x600 instead anyway.')
    G_GAP = args.gap
    G_OFFSET = args.offset
    G_ITER = args.iter
    G_STYLE = args.type
    G_BOLD = args.bold
    G_TEXT = args.text


def readimg(filename, mode):
    raw_data = np.fromfile(filename, dtype=np.uint8)  # 先用numpy把图片文件存入内存：raw_data，把图片数据看做是纯字节数据
    img = cv2.imdecode(raw_data, mode)  # 从内存数据读入图片
    return img


def new_resize(src, dsize: tuple):
    d_w, d_h = dsize
    s_h, s_w = src.shape[:2]

    # print(d_h, d_w, s_h, s_w)
    if d_h < s_h * d_w / s_w:
        scale_percent = d_w / s_w
    else:
        scale_percent = d_h / s_h
    # print(src.shape)
    canvas = np.zeros((d_h, d_w, 3), dtype="uint8")
    # scale_percent = max(dsize) / min(src.shape[:2])
    # scale_percent = 60  # percent of original size
    # print(scale_percent)
    width = int(src.shape[1] * scale_percent)
    height = int(src.shape[0] * scale_percent)
    dim = (width, height)
    # print(dim)
    resized = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)
    # print(resized.shape, dsize)
    if width == d_w:
        start_point = int((height - d_h) / 2)
        canvas[:, :] = resized[start_point:start_point + d_h, :]
    else:
        start_point = int((width - d_w) / 2)
        canvas[:, :] = resized[:, start_point:start_point + d_w]
    # print(canvas.shape[:2])
    return canvas
    return resized


def split_box(src: list, iters: int, gap: int, offset_rate: float) -> list:
    len_src = len(src)
    if len_src >= iters:
        return src
    if not 0 <= offset_rate <= 1:
        print('Error: move_rate must be a float which is lower than 1 and bigger than 0!')
        return []
    res = []
    for k, i in enumerate(src):
        try:
            if random.choice((True, False)):
                spilt_point = i[0] + random.randint(int((i[2] - i[0]) * (0.5 - offset_rate / 2)),
                                                    int((i[2] - i[0]) * (0.5 + offset_rate / 2)))
                res.append((i[0], i[1], spilt_point, i[3]))
                len_src = len(src) + len(res)
                if len_src >= iters:
                    src.pop(k)
                    res.append((spilt_point + gap, i[1], i[2], i[3]))
                    return res + src
                res.append((spilt_point + gap, i[1], i[2], i[3]))
            else:
                spilt_point = i[1] + random.randint(int((i[3] - i[1]) * (0.5 - offset_rate / 2)),
                                                    int((i[3] - i[1]) * (0.5 + offset_rate / 2)))
                res.append((i[0], i[1], i[2], spilt_point))
                len_src = len(src) + len(res)
                if len_src >= iters:
                    src.pop(k)
                    res.append((i[0], spilt_point + gap, i[2], i[3]))
                    return res + src
                res.append((i[0], spilt_point + gap, i[2], i[3]))
        except Exception as e:
            print(e)
            return res
    return split_box(res, iters, gap, offset_rate)


def split_box2(src: list, iters: int, gap: int, offset_rate: float) -> list:
    while 1:
        if len(src) >= iters:
            return src
        else:
            area = random.sample(src, 1)[0]
            index = src.index(area)
            src.pop(index)
            try:
                if random.choice((True, False)):
                    spilt_point = area[0] + random.randint(int((area[2] - area[0]) * (0.5 - offset_rate / 2)),
                                                           int((area[2] - area[0]) * (0.5 + offset_rate / 2)))
                    src.append((area[0], area[1], spilt_point, area[3]))
                    src.append((spilt_point + gap, area[1], area[2], area[3]))
                else:
                    spilt_point = area[1] + random.randint(int((area[3] - area[1]) * (0.5 - offset_rate / 2)),
                                                           int((area[3] - area[1]) * (0.5 + offset_rate / 2)))
                    src.append((area[0], area[1], area[2], spilt_point))
                    src.append((area[0], spilt_point + gap, area[2], area[3]))
            except Exception as e:
                print(e)
                return src


def dir2image_type_1():
    imgs = []
    for i in os.listdir(convert_path(images_dir_path)):
        if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg'):
            imgs.append(os.path.join(convert_path(images_dir_path), convert_path(i)))
    canvas = np.zeros((G_HEIGHT, G_WIDTH, 3), dtype="uint8")
    vimgs = []
    himgs = []
    nimgs = []
    for i in imgs:
        width, height = Image.open(i).size
        whrate = width / height
        if whrate < 0.7:
            vimgs.append(i)
        elif whrate > 1.3:
            himgs.append(i)
        else:
            nimgs.append(i)
    print(len(vimgs), len(himgs), len(nimgs))
    random.shuffle(vimgs)
    random.shuffle(himgs)
    random.shuffle(nimgs)
    iter_vimg = iter(vimgs)
    iter_himg = iter(himgs)
    iter_nimg = iter(nimgs)
    for box in split_box2([(0, 0, G_WIDTH, G_HEIGHT)], G_ITER, G_GAP, G_OFFSET):
        if (box[2] - box[0]) / (box[3] - box[1]) > 1.3:
            try:
                img1 = readimg(next(iter_himg), cv2.IMREAD_COLOR)
            except:
                try:
                    img1 = readimg(random.sample(nimgs, 1)[0], cv2.IMREAD_COLOR)
                except:
                    img1 = readimg(random.sample(vimgs, 1)[0], cv2.IMREAD_COLOR)
        elif (box[2] - box[0]) / (box[3] - box[1]) < 0.7:
            try:
                img1 = readimg(next(iter_vimg), cv2.IMREAD_COLOR)
            except:
                try:
                    img1 = readimg(random.sample(nimgs, 1)[0], cv2.IMREAD_COLOR)
                except:
                    img1 = readimg(random.sample(himgs, 1)[0], cv2.IMREAD_COLOR)
        else:
            try:
                img1 = readimg(next(iter_nimg), cv2.IMREAD_COLOR)
            except:
                try:
                    img1 = readimg(random.sample(himgs, 1)[0], cv2.IMREAD_COLOR)
                except:
                    img1 = readimg(random.sample(vimgs, 1)[0], cv2.IMREAD_COLOR)
        canvas[box[1]:box[3], box[0]:box[2]] = new_resize(img1, (box[2] - box[0], box[3] - box[1]))
    return canvas


def dir2image_type_2(bold: int):
    imgs = []
    for i in os.listdir(convert_path(images_dir_path)):
        if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg'):
            imgs.append(os.path.join(convert_path(images_dir_path), convert_path(i)))
    canvas = np.zeros((G_HEIGHT, G_WIDTH, 3), dtype="uint8")
    canvas[:, :] = (255, 255, 255)
    vimgs = []
    himgs = []
    nimgs = []
    for i in imgs:
        width, height = Image.open(i).size
        whrate = width / height
        if whrate < 0.7:
            vimgs.append(i)
        elif whrate > 1.3:
            himgs.append(i)
        else:
            nimgs.append(i)
    print(len(vimgs), len(himgs), len(nimgs))
    random.shuffle(vimgs)
    random.shuffle(himgs)
    random.shuffle(nimgs)
    if len(vimgs) > 0:
        main_image = vimgs.pop()
    elif len(nimgs) > 0:
        main_image = nimgs.pop()
    else:
        main_image = himgs.pop()
    iter_vimg = iter(vimgs)
    iter_himg = iter(himgs)
    iter_nimg = iter(nimgs)
    main_img = readimg(main_image, cv2.IMREAD_COLOR)
    k = int(min(main_img.shape[:2]) / 4) * 2 + 1
    canvas[:, :] = new_resize(cv2.GaussianBlur(main_img, ksize=(k, k), sigmaX=0, sigmaY=0), (G_WIDTH, G_HEIGHT))
    # canvas[bold + 3:G_HEIGHT - bold + 3, int(G_WIDTH / 2) + bold + 3:G_WIDTH - bold + 3] = [30, 30, 30]
    # canvas = add_shadow(canvas, (int(G_WIDTH / 2) + bold, bold, G_WIDTH - bold, G_HEIGHT - bold), 10)
    canvas[bold:G_HEIGHT - bold, int(G_WIDTH / 2) + bold:G_WIDTH - bold] = new_resize(main_img,
                                                                                      (int(G_WIDTH / 2) - bold * 2,
                                                                                       G_HEIGHT - bold * 2))
    for box in split_box2([(bold, bold, int(G_WIDTH / 2) - bold, G_HEIGHT - bold)], G_ITER, G_GAP, G_OFFSET):
        if (box[2] - box[0]) / (box[3] - box[1]) > 1.3:
            try:
                img1 = readimg(next(iter_himg), cv2.IMREAD_COLOR)
            except:
                try:
                    img1 = readimg(random.sample(nimgs, 1)[0], cv2.IMREAD_COLOR)
                except:
                    img1 = readimg(random.sample(vimgs, 1)[0], cv2.IMREAD_COLOR)
        elif (box[2] - box[0]) / (box[3] - box[1]) < 0.7:
            try:
                img1 = readimg(next(iter_vimg), cv2.IMREAD_COLOR)
            except:
                try:
                    img1 = readimg(random.sample(nimgs, 1)[0], cv2.IMREAD_COLOR)
                except:
                    img1 = readimg(random.sample(himgs, 1)[0], cv2.IMREAD_COLOR)
        else:
            try:
                img1 = readimg(next(iter_nimg), cv2.IMREAD_COLOR)
            except:
                try:
                    img1 = readimg(random.sample(himgs, 1)[0], cv2.IMREAD_COLOR)
                except:
                    img1 = readimg(random.sample(vimgs, 1)[0], cv2.IMREAD_COLOR)
        canvas[box[1]:box[3], box[0]:box[2]] = new_resize(img1, (box[2] - box[0], box[3] - box[1]))
    print(output_image_path)
    return canvas


def which_font(words: str):
    for word in words:
        if u"\u30a0" <= word <= u"\u30ff":
            return 'jp'
        if u"\u2e80" <= word <= u"\u2eff":
            return 'ko'
        if u'\u4e00' <= word <= u'\u9fa5':
            return 'ch'
    return 'eng'


def get_font_path(src: str, type: str):
    font_dict = {'ch': ['ZhiMangXing-Regular.ttf', 'MaShanZheng-Regular.ttf'],
                 'jp': [],
                 'ko': [],
                 'eng': ['MaShanZheng-Regular.ttf'],
                 'other': ['Marks.ttf', 'ttes.ttf', 'Stencils.ttf']}
    if type == 'default':
        font_type = which_font(src)
        font_path = convert_path(os.path.join(G_FONT_DIR, random.sample(font_dict[font_type], 1)[0]))
    else:
        font_path = convert_path(os.path.join(G_FONT_DIR, random.sample(font_dict[type], 1)[0]))
    return font_path


def add_text(src: Image, text: str, type: str, position: tuple, font_size: int, style: str):
    if style == 'default':
        setFont = ImageFont.truetype(get_font_path(text, type), font_size)
        draw = ImageDraw.Draw(src)
        draw.text(position, text, font=setFont, fill="#ffffff", spacing=0, align='left')
        return src
    elif style == 'v':
        setFont = ImageFont.truetype(get_font_path(text, type), font_size)
        draw = ImageDraw.Draw(src)
        w, h = position
        font_height = 40
        for k, v in enumerate(text):
            draw.text((w + 2, h + 2 + k * font_height), v, font=setFont, fill="#000000", spacing=0, align='left')
            draw.text((w, h + k * font_height), v, font=setFont, fill="#ffffff", spacing=0, align='left')
        return src
    else:
        ...


def add_shadow(src, area: tuple, bold: int):
    ww = area[2] - area[0]
    hh = area[3] - area[1]
    for ii in range(hh):
        for jj in range(bold):
            x = area[0] + ww + jj
            y = area[1] + ii + bold
            # src[y, x] = (0, 0, 0)
            src[y, x] = (
                            int(src[y, x][0] * (jj / bold)),
                            int(src[y, x][1] * (jj / bold)),
                            int(src[y, x][2] * (jj / bold))
                        )
    for ii in range(ww-bold):
        for jj in range(bold):
            x = area[0] + ii + bold
            y = area[1] + hh + jj
            src[y, x] = (
                int(src[y, x][0] * (jj / bold)),
                int(src[y, x][1] * (jj / bold)),
                int(src[y, x][2] * (jj / bold))
            )
    return src


if __name__ == '__main__':
    get_argument()
    if G_STYLE == 'default':
        cv2.imwrite(output_image_path, dir2image_type_1())
        print(output_image_path)
    elif G_STYLE == 'cover':
        cv2.imwrite(output_image_path, dir2image_type_2(G_BOLD))
        print(output_image_path)
    elif G_STYLE == 'cover_font':
        the_image = Image.fromarray(cv2.cvtColor(dir2image_type_2(G_BOLD), cv2.COLOR_BGR2RGB))
        the_image = add_text(the_image, G_TEXT, 'default', G_TEXT_POS, 40, 'v')
        # the_image = add_text(the_image, 'b', 'other', G_TEXT_POS, 30)
        the_image.save(output_image_path)
    elif G_STYLE == 'video_cover':
        from tqdm import tqdm
        import subprocess

        cap = cv2.VideoCapture()
        cap.open(images_dir_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(n_frames / fps / G_ITER / 2)

        with tempfile.TemporaryDirectory() as imgsdir:
            for i in tqdm(range(int(G_ITER * 2)), desc='Processing'):
                # cmd = "ffmpeg -ss {} -i '{}' -s {}x{} -frames 1 {}/{:0>2d}.png -v quiet" \
                #     .format(int(cutting_time) * (i + 1), video_path, S_Width, S_Height, imgsdir, i + 1)
                # os.system(cmd)

                res = subprocess.check_output(
                    ['ffmpeg', '-ss', '{}'.format(int(frame_interval) * (i + 1)), '-i', images_dir_path,
                     '-frames', '1', '{}/{:0>2d}.png'.format(imgsdir, i + 1)],
                    shell=False,
                    stderr=subprocess.STDOUT
                )
            images_dir_path = imgsdir
            cv2.imwrite(output_image_path, dir2image_type_2(G_BOLD))
            print(output_image_path)
