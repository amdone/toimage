import os
import cv2
import random
import argparse
import numpy as np
from PIL import Image

images_dir_path = './'
output_image_path = 'output.png'
G_WIDTH = 800
G_HEIGHT = 600
G_GAP = 0
G_OFFSET = 0
G_ITER = 4


def convert_path(path: str) -> str:
    return path.replace(r'\/'.replace(os.sep, ''), os.sep)


def get_argument():
    parser = argparse.ArgumentParser(description='必须给定一个目录,里面包含至少一张图片')
    parser.add_argument('images_dir_path', type=str, help='图片目录')
    parser.add_argument("-s", "--size", type=str, help="输出图片的大小，默认为800x600", default='800x600')
    parser.add_argument("-o", "--output", type=str, help='输出图片路径', default='./output.png')
    parser.add_argument("-g", "--gap", type=int, help='图片之间的间隙', default=0)
    parser.add_argument("-f", "--offset", type=float, help='随机拼接图片时的图片位置偏移概率', default=0)
    parser.add_argument("-i", "--iter", type=int, help='迭代次数', default=1)

    args = parser.parse_args()
    global images_dir_path, output_image_path, G_HEIGHT, G_WIDTH, G_GAP, G_ITER, G_OFFSET
    images_dir_path = convert_path(os.path.join(os.path.abspath('./'), args.images_dir_path))
    output_image_path = convert_path(os.path.join(os.path.abspath('./'), args.output))
    try:
        G_WIDTH = int(args.size.split('x')[0])
        G_HEIGHT = int(args.size.split('x')[1])
    except:
        print('Check your input of size, is it like this "800x600"?\nif not,I will use 800x600 instead anyway.')
    G_GAP = args.gap
    G_OFFSET = args.offset
    G_ITER = args.iter


def readimg(filename, mode):
    raw_data = np.fromfile(filename, dtype=np.uint8)  # 先用numpy把图片文件存入内存：raw_data，把图片数据看做是纯字节数据
    img = cv2.imdecode(raw_data, mode)  # 从内存数据读入图片
    return img


def cvopt():
    im = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)
    # print(im[0, 0])
    for i in range(300):
        for j in range(300):
            if abs(i - j) < random.randint(1, 300):
                im[i, j] = [255, 255, 255, 0]
                im[i, 300 - j] = [255, 255, 255, 0]
    cv2.imwrite('cv.png', im)


def new_image():
    canvas = np.zeros((300, 300, 3), dtype="uint8")
    cv2.line(canvas, (0, 0), (299, 299), (0, 255, 0))
    cv2.putText(canvas, '123', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite('line.png', canvas)


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
            area = random.sample(src,1)[0]
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



def fetch_face_pic(img, face_cascade):
    # 将图像灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸检测

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=10, minSize=(30, 30), flags=0)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, x + h), (255, 0, 0), 2)
        face_re = img[y:y + h, x:x + h]
        face_re_g = gray[y:y + h, x:x + h]
        # eyes = eye_cascade.detectMultiScale(face_re_g)
        # for (ex, ey, ew, eh) in eyes:
        #     cv.rectangle(face_re, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def face():
    # openCV里已经训练好的haar人脸检测器
    face_cascade = cv2.CascadeClassifier(
        'E:\\Python\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

    path_jaffe = 'F:\\PythonProjects\\PythonSpiders\\imgsOpt\\imgs'
    path_jaffe2 = 'F:\\PythonProjects\\PythonSpiders\\imgsOpt\\outimgs'

    # 遍历处理文件里所有人脸图像
    for file in os.listdir(path_jaffe):
        jaffe_pic = os.path.join(path_jaffe, file)
        img = cv2.imread(jaffe_pic)
        crop2 = fetch_face_pic(img, face_cascade)

        # cv2.imshow("Faces found", img)
        # cv2.imshow('Crop image', crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 将图像缩放到64*64大小
        resized_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        # 保存图像
        cv2.imwrite(jaffe_pic + '_out.jpg', resized_img)


if __name__ == '__main__':
    get_argument()
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
    cv2.imwrite(output_image_path, canvas)
    print(output_image_path)
    # cv2.imshow('result', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
