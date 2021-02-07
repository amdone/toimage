import os
import platform
import json
import tempfile
import subprocess
import argparse

from tqdm import tqdm
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

parser = argparse.ArgumentParser(description='必须给定一个视频文件的路径')
# type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('filepath', type=str, help='文件路径')
parser.add_argument("-r", "--row", type=int, help="行数")
parser.add_argument("-c", "--column", type=int, help="列数")
parser.add_argument("-s", "--size", type=str, help="行x列")
parser.add_argument("-w", "--width", type=int, help="图片宽度")
parser.add_argument("-i", "--info", type=int, help="头部信息")
parser.add_argument("-o", "--output", type=str, help="输出文件路径")

args = parser.parse_args()

# 获得传入的参数
video_path = os.path.join(os.path.abspath('./'), args.filepath)
print(video_path)

Output = './output.png'
video_name = ''
# FontPath = './OPPOSans-B-2.ttf'
FontPath = os.path.join(os.path.dirname(__file__), 'OPPOSans-B-2.ttf')
if platform.system() == 'Windows':
    video_name = args.filepath.split('\\')[-1]
elif platform.system() == 'Linux':
    video_name = args.filepath.split('/')[-1]
Output = ''.join(video_name.split('.')[:-1]).__add__('.png')
print((video_name, Output))


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
        if streams['width']:
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
            video_name,
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

Row = 6
Column = 5
S_Width = 300
Head_Size = 80



if args.row:
    Row = args.row
if args.column:
    Column = args.column
if args.width:
    S_Width = args.width
if args.output:
    Output = args.output


S_Height = int(VIDEO_HEIGHT * 1.0 / VIDEO_WIDTH * S_Width)
cutting_time = ffprobe.video_time_length_second() / (Row * Column)
end_time = ffprobe.video_time_length_second() - 1
FPS = Row * Column / ffprobe.video_time_length_second()
print((S_Height, S_Width, FPS))


# ff = FFmpeg(
#     inputs={video_path: None},
#     # outputs={'e:/test/cutimgs/%03d.png': ['-s', '{}x{}'.format(S_Width, S_Height), '-vf', 'fps={}'.format(FPS)]}
#     outputs={'e:/test/cutimgs/%03d.png': ['-ss', '6000', '-s', '{}x{}'.format(S_Width, S_Height), 'frames', '1', '-f', 'p_w_picpath2']}
#     # '''ffmpeg -y -i  beijing-480p.mp4 -ss 6000 -s 320x180 -frames 1 -f p_w_picpath2 result.jpg'''
# )

def Add_Time_To_Img(FilePath, Img_Height, Seq):
    # 设置文字颜色
    fillColor_White = "#ffffff"
    fillColor_Black = "#000000"
    fillColor_Gray = "#005500"
    # img = Image.new('RGB', (1200, 1500), (220, 220, 220))
    img = Image.open(FilePath)
    draw = ImageDraw.Draw(img)
    # 选择文字字体和大小
    setFont = ImageFont.truetype(FontPath, 15)
    # 写入文字
    draw.text((6, Img_Height - 19), Seq, font=setFont, fill=fillColor_Black)
    draw.text((5, Img_Height - 20), Seq, font=setFont, fill=fillColor_White)
    img.save(FilePath)


def Out_Image(Seq, Images_Dir):
    # 设置文字颜色
    fillColor_White = "#ffffff"
    fillColor_Black = "#000000"
    fillColor_Gray = "#005500"
    img = Image.new('RGB', (Column * S_Width, Head_Size + Row * S_Height), (220, 220, 220))
    draw = ImageDraw.Draw(img)
    # 选择文字字体和大小
    setFont = ImageFont.truetype(FontPath, 20)
    # 写入文字
    draw.text((10, 10), Seq, font=setFont, fill=fillColor_Black)
    # draw.text((10, 10), Seq, font=setFont, fill=fillColor_White)
    count = 0
    imgs = os.listdir(Images_Dir)
    for r in range(Row):
        for c in range(Column):
            try:
                this_img = os.path.join(Images_Dir, imgs[count])
                Add_Time_To_Img(this_img, S_Height, format_time(int(cutting_time) * (count + 1)))
                fromImage = Image.open(this_img)
                img.paste(fromImage, (c * S_Width, Head_Size + r * S_Height))
                # print(os.path.join('e:/test/cutimgs', imgs[count]))
                count += 1
            except:
                setFont = ImageFont.truetype(FontPath, 20)
                draw.text(((Column-0.65) * S_Width, Head_Size + (Row-0.6) * S_Height),
                          'The End', font=setFont, fill=fillColor_Black)
    img.save(Output)


with tempfile.TemporaryDirectory() as imgsdir:
    print(imgsdir)
    for i in tqdm(range(Row * Column), desc='Processing'):
        # cmd = "ffmpeg -ss {} -i '{}' -s {}x{} -frames 1 {}/{:0>2d}.png -v quiet" \
        #     .format(int(cutting_time) * (i + 1), video_path, S_Width, S_Height, imgsdir, i + 1)
        # os.system(cmd)
        res = subprocess.check_output(
            ['ffmpeg', '-ss', '{}'.format(int(cutting_time)*(i+1)), '-i', video_path, '-s', '{}x{}'.format(S_Width, S_Height),
             '-frames', '1', '{}/{:0>2d}.png'.format(imgsdir, i + 1)],
            shell=False,
            stderr=subprocess.STDOUT
        )
        # print('\rprocess: {:.2%}'.format((i+1) / (Row * Column)))
    # cmd = "ffmpeg -ss {} -i '{}' -s {}x{} -frames 1 {}/{:0>2d}.png -v quiet" \
    #     .format(int(end_time), video_path, S_Width, S_Height, imgsdir, 9999999)
    # os.system(cmd)
    res = subprocess.check_output(
        ['ffmpeg', '-ss', '{}'.format(int(end_time)), '-i', video_path, '-s', '{}x{}'.format(S_Width, S_Height),
         '-frames', '1', '{}/{:0>2d}.png'.format(imgsdir, 9999999)],
         shell=False, stderr=subprocess.STDOUT
    )
    # res = subprocess.check_output(
    #     ['ffmpeg', '-ss', '{}'.format(int(cutting_time)*(i+1)), '-i', video_path, '-s', '{}x{}'.format(S_Width, S_Height),
    #      '-frames', '1', './cutimgs/000{}.png'.format(i + 1)],
    #     shell=False,
    #     stderr=subprocess.STDOUT
    # )
    Out_Image(ffprobe.video_info_str(), imgsdir)
