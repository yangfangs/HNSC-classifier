from histolab.slide import Slide
import os
from histolab.tiler import RandomTiler
from histolab.tiler import GridTiler
from fastai.vision.all import *
import matplotlib.pyplot as plt
import warnings
import pandas as pd

warnings.simplefilter(action='ignore')
os.environ["OMP_NUM_THREADS"] = "1"


# TILE_SIZE = (224, 224)
# LEVEL = 0
# BASE_RESULT_PATH = "/Users/fangy/work/HNSC/test_PIL"


def extract_grid_slide(svs_path, tile_size, LEVEL, BASE_RESULT_PATH):
    """
    extract tiles
    :param svs_path: WSI path
    :param tile_size: tiles size
    :param LEVEL: extarct tiles form level
    :param BASE_RESULT_PATH: outpath
    :return: result dir
    """
    # svs_path = "/Users/fangy/work/HNSC/data/00b2dd2a-95f9-4e87-aef1-473524725b4c/TCGA-BA-7269-01A-01-TS1.44d70d72-41bc-4a13-9b44-6cbabe7f9ef2.svs"
    # svs_path = "/Users/fangy/work/HNSC/data/4b8dc6ea-de14-4ff7-ad9b-10b77a7e3e33/TCGA-CV-7245-11A-01-TS1.30e6a59b-fdc9-4976-9040-e1a851232b1b.svs"
    # img_path = '/Users/fangy/work/HNSC/data/4c650817-442e-4200-84cd-942b1378aa1c/TCGA-BB-4223-01A-01-BS1.7d09ad3d-016e-461a-a053-f9434945073b.svs'
    TILE_SIZE = (tile_size, tile_size)
    process_path = os.path.join(BASE_RESULT_PATH, "Extract_tiles")

    nhsc_slide = Slide(svs_path, processed_path=process_path)

    grid_tiles_extractor = GridTiler(
        tile_size=TILE_SIZE,
        level=LEVEL,
        check_tissue=True,  # default
        pixel_overlap=0,  # default
        prefix="",  # save tiles in the "grid" subdirectory of slide's processed_path
        suffix=".png"  # default
    )

    ex_res = grid_tiles_extractor.locate_tiles(
        slide=nhsc_slide,
        scale_factor=64,
        alpha=64,
        outline="red",
    )
    grid_tiles_extractor.extract(nhsc_slide)
    return process_path


# def label_func(f=''):
#     return f[:-4].split('_')[-1]


def predict(svs_path, model_path_cancer, tile_size, result_path):
    """
    predict cancer/normal
    :param svs_path: WSI path
    :param model_path_cancer: deep learn model path
    :param tile_size: tile size
    :param result_path: path of result
    :return: predicted result
    """

    # model_path_cancer = "/Users/fangy/work/HNSC/test_PIL/learn.pkl"
    # print("loading func")
    base_path = os.path.dirname(__file__)
    sys.path.append(base_path)
    # label_func()
    learn = load_learner(model_path_cancer)

    path = Path(svs_path)
    files = get_image_files(path, recurse=True)

    # learn.predict(files[0])

    test_dl = learn.dls.test_dl(files)  # Create dataloader
    preds, _, dec_preds = learn.get_preds(dl=test_dl, with_decoded=True)
    preds_num = preds.numpy()
    dec_preds_num = dec_preds.numpy()

    cls_name = learn.dls.vocab
    cancer_index = np.where(dec_preds_num == 0)[0]
    files_stage = files[cancer_index.tolist()]
    cancer_prob = preds_num[cancer_index.tolist(), 0]
    res = {'tile': [x for x in files],
           'pixel': [tile_size] * len(dec_preds_num),
           'classification': [cls_name[x] for x in dec_preds_num],
           'probability': [preds_num[x, dec_preds_num.tolist()[x]] for x in range(len(dec_preds_num.tolist()))]}

    df = pd.DataFrame(res)

    return df


def predict_stage(model_path_stage, df, title_stage, title_prob):
    """
    predcit stage
    :param model_path_stage: deep learn model path
    :param df: cancer/normal predicted result
    :param title_stage: stage titles
    :param title_prob:  probability of tiles
    :return: predicted result
    """
    files_stage = df[df['classification'] == 'cancer']['tile']
    # model_path_stage = "/Users/fangy/work/HNSC/test_PIL/learn_S.pkl"
    learn_stage = load_learner(model_path_stage)
    test_dl_s = learn_stage.dls.test_dl(files_stage)  # Create a test dataloader
    cls_stage_name = learn_stage.dls.vocab
    preds_s, _, dec_preds_s = learn_stage.get_preds(dl=test_dl_s, with_decoded=True)
    preds_s_num = preds_s.numpy()
    dec_preds_s_num = dec_preds_s.numpy().tolist()
    # prob = preds_s_num[3,2]
    row_list = list(range(len(dec_preds_s_num)))
    prob_stage = [preds_s_num[x, dec_preds_s_num[x]] for x in row_list]
    cals_stage = [cls_stage_name[x] for x in dec_preds_s_num]
    res_stage = {'tile': files_stage,
                 title_stage: cals_stage,
                 title_prob: prob_stage}
    df_stage = pd.DataFrame(res_stage)
    df_combine = pd.merge(df, df_stage, on='tile', how='left')
    return df_combine

