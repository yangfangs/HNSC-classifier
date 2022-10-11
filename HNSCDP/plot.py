import os
import random
import PIL
import matplotlib as mpl
import scipy
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy
import matplotlib.pyplot as plt
import pylab as pl
import openslide
from fastai.data.transforms import get_image_files
from fastai.vision.all import *
from openslide.deepzoom import DeepZoomGenerator
import io


# from colour import Color
# result_path = '/Users/fangy/work/HNSC/test_PIL'
#
# os.chdir(result_path)


# def get_color(num):
#     green = Color("green")
#     colors = list(green.range_to(Color("red"), num))
#     return colors


def plot_bar(n):
    ''' plot heatmap bar'''
    c1 = 'green'  # blue
    c2 = 'red'  # green
    # n = 100

    fig, ax = plt.subplots(figsize=(6, 2))
    plt.title('Probability')
    for x in range(n + 1):
        ax.axvline(x / 100, color=colorFader(c1, c2, x / n), linewidth=10)
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    plt.xticks(fontsize=10)

    return fig


def read_coordinate(path):
    """
    read coordinate from tiles
    :param path:  path of tiles
    :return: coordinate
    """
    # path = "/Users/fangy/work/HNSC/test_PIL/Extract"
    # path = Path("/Users/fangy/work/HNSC/test_PIL/Extract2/")
    # files = get_image_files(path, recurse=True)
    files = path
    coord = [tuple(map(int, x.name.split('_')[-1].split('.')[0].split('-'))) for x in files]
    return coord


def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    """
    define the color range
    :param c1: star
    :param c2: stop
    :param mix: 0
    :return: color range
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def resize_pos(trans, src_size, tar_size):
    """
    resize position
    :param trans: tarnsform coord
    :param src_size: source size
    :param tar_size: target size
    :return: coord
    """
    x1 = trans[0]
    y1 = trans[1]
    x11 = trans[2]
    y11 = trans[3]
    w1 = src_size[0]
    h1 = src_size[1]
    w2 = tar_size[0]
    h2 = tar_size[1]
    y2 = (h2 / h1) * y1
    x2 = (w2 / w1) * x1
    y22 = (h2 / h1) * y11
    x22 = (w2 / w1) * x11
    return (x2, y2, x22, y22)


def get_text(col_names, df_row):
    """
    get text form predicted result
    :param col_names: col names
    :param df_row: row of data frame
    :return:
    """
    summary_list = []
    summary_list.append("Id: " + str(df_row.name))
    if "classification" in col_names:
        summary_list.append("Classification: " + df_row.classification)
    if "probability" in col_names:
        summary_list.append("Prob.: " + str(df_row.probability)[:4])
    if "stage" in col_names:
        summary_list.append("Stage: " + df_row.stage)
    if "stage_probability" in col_names:
        summary_list.append("S_prob.: " + str(df_row.stage_probability)[:4])
    if "TNM_system_T" in col_names:
        summary_list.append("T: " + df_row.TNM_system_T)
    if "T_probability" in col_names:
        summary_list.append("T_prob.: " + str(df_row.T_probability)[:4])
    if "TNM_system_M" in col_names:
        summary_list.append("M: " + df_row.TNM_system_M)
    if "M_probability" in col_names:
        summary_list.append("T_prob.: " + str(df_row.M_probability)[:4])
    if "TNM_system_N" in col_names:
        summary_list.append("N: " + df_row.TNM_system_N)
    if "N_probability" in col_names:
        summary_list.append("N_prob.: " + str(df_row.N_probability)[:4])
    return "\n".join(summary_list)


def heatmap_svs(img_path, df_combine, out_path, method):
    """
    plot heatmap
    :param img_path: read image path
    :param df_combine: predict result
    :param out_path: output path
    :param method: method
    :return: NA
    """
    # img_path = '/Users/fangy/work/HNSC/data/4c650817-442e-4200-84cd-942b1378aa1c/TCGA-BB-4223-01A-01-BS1.7d09ad3d-016e-461a-a053-f9434945073b.svs'
    slide = openslide.open_slide(img_path)
    slide.dimensions
    # slide_thumbnail = slide.get_thumbnail(slide.dimensions)
    slide_thumbnail = slide.get_thumbnail((1000, 1000))
    draw = ImageDraw.Draw(slide_thumbnail)
    w, h = slide_thumbnail.size

    coord = read_coordinate(df_combine['tile'])
    # cols = get_color(100)
    prob = []
    for each in range(len(coord)):
        prob.append(random.uniform(0.8, 1))

    c1 = 'green'  # blue
    c2 = 'red'  # green
    n = len(coord)

    for line in range(n):
        # draw.rectangle(xy=resize_pos(coord[line],slide.dimensions,slide_thumbnail.size), fill=cols[int(round(prob[line],2)*100 -1)].get_hex(), outline=None, width=2)
        draw.rectangle(xy=resize_pos(coord[line], slide.dimensions, slide_thumbnail.size),
                       fill=colorFader(c1, c2, (round(prob[line], 2) * 100) / 100), outline=None, width=2)
    # slide_thumbnail.show()

    # image.show()
    fig = plot_bar(n=100)
    fig2 = fig2img(fig)
    # fig2.show()
    w2, h2 = fig2.size

    W = w + w2 + 50
    H = h + h2 + 50
    image = Image.new(mode="RGB", size=(W, H), color="white")
    draw1 = ImageDraw.Draw(image)
    image.paste(im=slide_thumbnail, box=(5, 90))

    image.paste(im=fig2, box=(w + 6, 90))

    # slide_thumbnail.save(os.path.join(out_path, 'cancer_heatmap.png'))
    summary_list = []

    summary_list.append("total tiles: " + str(df_combine.shape[0]))

    summary = df_combine[method].value_counts()

    for idx, row in summary.items():
        # print(idx, row)
        summary_list.append(idx + ": " + str(row) + ", " + str(round(row / df_combine.shape[0], 3) * 100) + "%")

    text = "\n".join(summary_list)

    font2 = ImageFont.truetype(font='FreeSerif.ttf', size=20)
    font_summary = ImageFont.truetype(font='FreeSerif.ttf', size=20)

    draw1.text(xy=(w + (W - w) // 2 - 80, h2 + 200),
               text="Summary: ", fill='black', font=font_summary, stroke_width=1,
               stroke_fill="black")

    draw1.text(xy=(w + (W - w) // 2 - 80, h2 + 240),
               text=text, fill='black', font=font2)
    # image.show()
    if method == "classification":
        save_name = "cancer_heatmap.png"
    elif method == "stage":
        save_name = "stage_heatmap.png"
    elif method == "TNM_system_T":
        save_name = "TNM_system_T_heatmap.png"
    elif method == "TNM_system_M":
        save_name = "TNM_system_M_heatmap.png"
    elif method == "TNM_system_N":
        save_name = "TNM_system_N_heatmap.png"

    image.save(os.path.join(out_path, save_name))


def read_svs(img_path, df_combine, out_path):
    """
    read svs format image
    :param img_path: image path
    :param df_combine: predicted result
    :param out_path: output path
    :return: NA
    """
    w_each = 224
    h_each = 224
    # img_path = '/Users/fangy/work/HNSC/data/00b2dd2a-95f9-4e87-aef1-473524725b4c/TCGA-BA-7269-01A-01-TS1.44d70d72-41bc-4a13-9b44-6cbabe7f9ef2.svs'
    slide = openslide.open_slide(img_path)
    slide_thumbnail = slide.get_thumbnail((1000, 1000))
    draw = ImageDraw.Draw(slide_thumbnail)
    w, h = slide_thumbnail.size
    font = ImageFont.truetype(font='FreeSerif.ttf', size=10)
    df_combine_top = df_combine[:8]
    for row_index, row in df_combine_top.iterrows():
        # print(row_index, row)
        coord = tuple(map(int, row[0].name.split('_')[-1].split('.')[0].split('-')))
        # draw.rectangle(xy=coord, fill=None, outline="red", width=1)
        coord_xy = resize_pos(coord, slide.dimensions, slide_thumbnail.size)
        draw.rectangle(xy=coord_xy,
                       outline="red", width=1)
        # font = ImageFont.truetype(font='PingFang.ttc', size=5)
        draw.text(xy=coord_xy[:2], text=str(row_index), fill='green', font=font)

    W = 1600
    H = 1700 + h
    image = Image.new(mode="RGB", size=(W, H), color="white")
    image.paste(im=slide_thumbnail, box=((W - w) // 2, 30))
    draw1 = ImageDraw.Draw(image)
    # im = Image.open(row[0])
    # photo = im.resize((w_each, h_each))
    focus_point = [0.125 * W, 0.25 * (H - h) + h]
    start_point = [focus_point[0] - 0.5 * w_each, focus_point[1] - 0.5 * h_each]
    count = 0
    col_names = df_combine_top.columns.tolist()
    font2 = ImageFont.truetype(font='FreeSerif.ttf', size=30)
    for i in range(0, 2):
        for k in range(0, 4):
            df_row = df_combine_top.loc[count]
            im = Image.open(df_row[0])
            photo = im.resize((w_each, h_each))
            image.paste(photo, (int(start_point[0] + (k * W / 4)), int(start_point[1] + 0.45 * i * (H - h))))

            # text = 'ID:{0}\nClassify: {1}\nProb.: {2}\nStatge: {3}\nProb.: {4}\nT: T2\nM: M0\nN: N0'.format(
            #     str(df_row.name), df_row[2], str(df_row[3])[:4], df_row[4], str(df_row[5])[:4])
            text = get_text(col_names, df_row)
            draw1.text(xy=(int(start_point[0] + (k * W / 4)), int(start_point[1] + 0.45 * i * (H - h) + h_each)),
                       #            # text='ID:{0}\nClassify: Cancer\nProb.: 0.98\nStatge: II\nT: T2\nM: M0\nN: N0'.format(str(count)) , fill=(255, 0, 0), font=font)
                       text=text, fill=(255, 0, 0), font=font2)
            count += 1
    # image.show()
    image.save(os.path.join(out_path, "summary.png"))
