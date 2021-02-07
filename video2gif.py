import os
import json
import tempfile
import subprocess
import argparse
import imageio

from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser(description='必须给定一个视频文件的路径')
# type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('filepath', type=str, help='文件路径')
parser.add_argument("-f", "--frames", type=int, help="帧率")
parser.add_argument("-c", "--counts", type=int, help="截取次数")
parser.add_argument("-s", "--second", type=str, help="行x列")
parser.add_argument("-w", "--width", type=int, help="图片宽度")
parser.add_argument("-i", "--info", type=int, help="头部信息")
parser.add_argument("-o", "--output", type=str, help="输出文件路径")

args = parser.parse_args()

# 获得传入的参数
video_path = os.path.join(os.path.abspath('./'), args.filepath)
print(video_path)


def format_time(tt):
    m, s = divmod(tt, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


class FFprobe:
    def __init__(self):
        self.filepath = ''
        self._video_info = {}

    def parse(self, filepath):
        self.filepath = filepath
        try:
            res = subprocess.check_output(
                ['ffprobe', '-i', self.filepath, '-print_format', 'json', '-show_format', '-show_streams', '-v',
                 'quiet'])
            res = res.decode('utf8')
            self._video_info = json.loads(res)
            # print('_video_info ',self._video_info)
        except Exception as e:
            print(e)
            raise Exception('获取视频信息失败')

    def get_json(self):
        return self._video_info

    def video_width_height(self):
        streams = self._video_info['streams'][0]
        streams2 = self._video_info['streams'][1]
        if streams['codec_type'] == 'video':
            return (streams['width'], streams['height'])
        else:
            return (streams2['width'], streams2['height'])

    def video_filesize(self, format='gb'):
        v_format = self._video_info['format']
        size = int(v_format['size'])
        kb = 1024
        mb = kb * 1024
        gb = mb * 1024
        tb = gb * 1024
        if size >= tb:
            return "%.1f TB" % float(size / tb)
        if size >= gb:
            return "%.1f GB" % float(size / gb)
        if size >= mb:
            return "%.1f MB" % float(size / mb)
        if size >= kb:
            return "%.1f KB" % float(size / kb)

    def video_full_frame(self):
        stream = self._video_info['streams'][0]
        return stream['nb_frames']

    def video_bit_rate(self):
        return int(self._video_info['format']['bit_rate']) / 1024

    def video_frame_rate(self):
        streams = self._video_info['streams'][0]
        streams2 = self._video_info['streams'][1]
        if streams['codec_type'] == 'video':
            return eval(streams['r_frame_rate'])
        else:
            return eval(streams2['r_frame_rate'])

    def video_time_length(self):
        v_format = int(float(self._video_info['format']['duration']))
        return format_time(v_format)

    def video_time_length_second(self):
        return int(float(self._video_info['format']['duration']))

    def video_info(self):
        item = {
            'path': self.filepath,
            'height_width': self.video_width_height(),
            'filesize': self.video_filesize(),
            'time_length': self.video_time_length()
        }
        # print('item = ', item)
        return item

    def video_info_str(self):
        res_str = 'Name: {}\nFileSize: {}  VideoSize:{}  Time: {}  Bit_rate: {:.2f}kbps  FPS: {:.2f}'.format(
            self.filepath.split('/')[-1],
            self.video_filesize(),
            self.video_width_height(),
            self.video_time_length(),
            self.video_bit_rate(),
            self.video_frame_rate()
        )
        return res_str


ffprobe = FFprobe()
ffprobe.parse(video_path)
video_info = ffprobe.get_json()
print(ffprobe.video_info_str())
print(video_info['format']['duration'])

VIDEO_WIDTH = ffprobe.video_width_height()[0]
VIDEO_HEIGHT = ffprobe.video_width_height()[1]
VIDEO_TIME = ffprobe.video_time_length()

Frames = 10
Counts = 5
S_Width = 300
Head_Size = 80
Second = 3
Output = '{}.gif'.format(args.filepath.split('.')[-2])
# FontPath = './OPPOSans-B-2.ttf'
FontPath = os.path.join(os.path.dirname(__file__), 'OPPOSans-B-2.ttf')
if args.frames:
    Frames = args.frames
if args.counts:
    Counts = args.counts
if args.width:
    S_Width = args.width
if args.output:
    Output = args.output
if args.second:
    Second = args.second

S_Height = int(VIDEO_HEIGHT * 1.0 / VIDEO_WIDTH * S_Width)
cutting_time = ffprobe.video_time_length_second() / (Counts + 1)
end_time = ffprobe.video_time_length_second() - 1
FPS = Counts / ffprobe.video_time_length_second()
print((S_Height, S_Width, FPS))

with tempfile.TemporaryDirectory() as imgsdir:
    print(imgsdir)
    img_count = 0
    for i in tqdm(range(Counts), desc='Processing'):
        cmd = 'ffmpeg -ss {} -t {} -i {} -s {}x{} -r {} {}/{:0>5d}.gif -v quiet' \
            .format(int(cutting_time) * (i + 1), Second, video_path, S_Width,
                    S_Height, Frames, imgsdir, img_count + 1)
        os.system(cmd)
        img_count += 1
    img_count = 0
    images = []
    filenames = sorted((fn for fn in os.listdir(imgsdir) if fn.endswith('.gif')))
    for filename in filenames:
        im = Image.open(os.path.join(imgsdir, filename))
        try:
            ic = 0
            while True:
                im.seek(ic)
                im.save(os.path.join(imgsdir, '{:0>5d}.png'.format(img_count)))
                ic += 1
                img_count += 1
        except:
            im.close()
            pass
    filenames = sorted((fn for fn in os.listdir(imgsdir) if fn.endswith('.png')))
    for filename in filenames:
        images.append(imageio.imread(os.path.join(imgsdir, filename)))
    imageio.mimsave(Output, images, 'GIF', duration=1 / Frames)
    try:
        cmd = 'gifscile -b -O2 {}'.format(Output)
    except:
        pass
