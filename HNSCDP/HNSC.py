import argparse
import os
import time
from HNSCDP.HNSC_predict import extract_grid_slide, predict, predict_stage
from HNSCDP.plot import heatmap_svs, read_svs

# arguments
parser = argparse.ArgumentParser(description='HNSC-classifier')

parser.add_argument("-i", "--image", dest='wsi_path', required=True, help="Path to a whole slide image")

parser.add_argument("-o", "--output", dest='output_name', default="output",
                    help="Name of the output file directory [default: `output/`]")

parser.add_argument("-p", "--tile_size", dest='tile_size', default=224, type=int,
                    help="The pixel width and height for tiles")

parser.add_argument("-l", "--level", dest='level', default=0, type=int,
                    help="Extract tiles form resolution of level")

parser.add_argument("-c", "--cancer", dest='cancer_model', default="",
                    help="The deep model path of cancer/normal classification")

parser.add_argument("-s", "--stage", dest='S_model', default="",
                    help="The deep model path of stage classification")

parser.add_argument("-t", "--tumor", dest='T_model', default="",
                    help="The deep model path of T classification (TNM Staging System)")

parser.add_argument("-n", "--nearby", dest='N_model', default="",
                    help="The deep model path of N classification (TNM Staging System)")

parser.add_argument("-m", "--metastasized", dest='M_model', default="",
                    help="The deep model path of M classification (TNM Staging System)")

args = parser.parse_args()


def main():

    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, args.output_name)
    os.makedirs(final_directory, exist_ok=True)
    print(final_directory)
    t_start = time.time()
    t = time.time()
    process_path = extract_grid_slide(svs_path=args.wsi_path,
                       tile_size=args.tile_size,
                       LEVEL= args.level,
                       BASE_RESULT_PATH=final_directory)

    print("Extract tiles finished")
    print(f'elapsed time: {time.time() - t:.4f}s')

    # process_path = "/Users/fangy/work/HNSC/test_PIL/output/Extract_tiles"
    # final_directory = "/Users/fangy/work/HNSC/test_PIL/output/"

    # label_func()

    t = time.time()
    # predicted cancer/normal
    res_predict = predict(svs_path=process_path,
                          model_path_cancer=args.cancer_model,
                          tile_size=args.tile_size,
                          result_path=final_directory)


    # res_predict = predict(svs_path=process_path,
    #                       model_path_cancer="/Users/fangy/work/HNSC/test_PIL/learn.pkl",
    #                       tile_size=224,
    #                       result_path=final_directory)

    print("predict cancer/normal tiles finished")
    # plot heatmap
    heatmap_svs(img_path=args.wsi_path,
                df_combine=res_predict,
                out_path=final_directory,
                method="classification")

    print("plot cancer heatmap finished")
    print(f'elapsed time: {time.time() - t:.4f}s')
    if args.S_model != "":
        t = time.time()
        # predict stage
        res_predict = predict_stage(model_path_stage=args.S_model, df=res_predict, title_stage="stage",
                                    title_prob="stage_probability")
        print("predict the tiles of stage finished")
        # plot heatmap
        heatmap_svs(img_path=args.wsi_path,
                    df_combine=res_predict,
                    out_path=final_directory,
                    method="stage")

        print("plot stage heatmap finished")

        print(f'elapsed time: {time.time() - t:.4f}s')
    if args.T_model != "":
        t = time.time()
        # predict TNM system (T)
        res_predict = predict_stage(model_path_stage=args.T_model, df=res_predict, title_stage="TNM_system_T",
                                    title_prob="T_probability")


        print("predict the tiles of primary tumor (T) finished")
        # plot heatmap
        heatmap_svs(img_path=args.wsi_path,
                    df_combine=res_predict,
                    out_path=final_directory,
                    method="TNM_system_T")

        print("plot stage heatmap finished")
        print(f'elapsed time: {time.time() - t:.4f}s')
    if args.M_model != "":
        t = time.time()
        # predict TNM system (M)
        res_predict = predict_stage(model_path_stage=args.M_model, df=res_predict, title_stage="TNM_system_M",
                                    title_prob="M_probability")
        print("predict the tiles of cancer metastasized (M) finished")
        # plot heatmap
        heatmap_svs(img_path=args.wsi_path,
                    df_combine=res_predict,
                    out_path=final_directory,
                    method="TNM_system_M")

        print("plot stage heatmap finished")
        print(f'elapsed time: {time.time() - t:.4f}s')
    if args.N_model != "":
        t = time.time()
        # predict TNM system (N)
        res_predict = predict_stage(model_path_stage=args.N_model, df=res_predict, title_stage="TNM_system_N",
                                    title_prob="N_probability")
        print("predict the tiles of nearby lymph nodes (N) finished")
        # plot heatmap
        heatmap_svs(img_path=args.wsi_path,
                    df_combine=res_predict,
                    out_path=final_directory,
                    method="TNM_system_N")

        print("plot stage heatmap finished")
        print(f'elapsed time: {time.time() - t:.4f}s')

    df_path = os.path.join(final_directory, "summary.csv")
    res_predict.to_csv(df_path)

    # plot result
    read_svs(img_path=args.wsi_path,
             df_combine=res_predict,
             out_path=final_directory)
    print("plot summary finished")
    print(f'total elapsed time: {time.time() - t_start:.4f}s')


if __name__ == '__main__':
    main()
