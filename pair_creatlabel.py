import tarfile
import os
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import json
import pandas as pd


def get_file_name(file_dir):
    # Parameter:
    # file_dir: the folder path of *.tar
    # return : [List]: all file path
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(file)
    return L


def un_tar(file_names, file_path):
    # Parameter:
    # file_names: the file name of *.tar
    # file_path: the save path about the unzipped files.(default: same path of *.tar)
    for name in tarfile.open(file_path + "\\" + file_names).getnames():
        tarfile.open(file_path + "\\" + file_names).extract(name, file_path)


def read_video_and_label(video_path, label_path, save_path, video_name, video_id):
    # video_path = r'F:\Xxy_Q\核查1\SP+H\20190909T161453__B_产科.mp4'
    img_save_path = save_path + 'image'
    mask_save_path = save_path + 'mask'
    csv_path = save_path + 'frame_label.csv'
    csv_list = []
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    if not os.path.exists(mask_save_path):
        os.mkdir(mask_save_path)

    videocapture = cv2.VideoCapture(video_path)
    nii_name = label_path + video_name + '_mp4_Label.nii.gz'
    nii_mask = nib.load(nii_name).get_fdata()  # 读取CT值形式nii

    ori_rate, framenumber = videocapture.get(5), videocapture.get(7)
    if nii_mask.shape[2] != int(framenumber):
        print('nii_mask.shape[2] != framenumber')
        return 0
    print('帧速:{}\t 帧数{}\n'.format(ori_rate, framenumber))
    json_name = label_path + video_name + '_mp4_Label.json'
    with open(json_name, 'r', encoding='utf-8') as f:
        json_info = json.load(f)
    all_frame_label = json_info['Models']['class FrameLabelModel * __ptr64']['FrameLabel']

    for single_frame in all_frame_label:
        num_frame = single_frame['FrameCount']
        single_frame_label = single_frame['Label']

        single_mask = nii_mask[:, :, num_frame].transpose()
        videocapture.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
        ret, frame = videocapture.read()
        if frame is None:
            print(video_id + '_' + str(int(num_frame)) + ': empty')
        else:
            csv_list.append([num_frame, single_frame_label])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 把GRAY图转换为BGR三通道图--BGR转灰度图
            frame = frame[54:, 528:1823]
            frame[0:434, 1175:] = 0
            frame[959:1003, 199:1048] = 0
            frame[:, 0:72] = 0
            single_mask = single_mask[54:, 528:1823]
            # save img mask
            img_name = video_id + '_' + str(int(num_frame)) + '.png'
            mask_name = video_id + '_' + str(int(num_frame)) + '_mask.png'
            cv2.imwrite(os.path.join(img_save_path, img_name), frame)
            cv2.imwrite(os.path.join(mask_save_path, mask_name), single_mask)

    # for n in range(int(framenumber)):
    #     videocapture.set(cv2.CAP_PROP_POS_FRAMES, n)
    #     ret, frame = videocapture.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 把GRAY图转换为BGR三通道图--BGR转灰度图
    #     frame = frame[54:, 528:1823]
    #     frame[0:434, 1175:] = 0
    #     frame[959:1003, 199:1048] = 0
    #     frame[:, 0:72] = 0
    # frame = cv2.resize(frame, (512, 384))
    # print('Read a new frame: ', success)
    csv_colname = ['frame_id', 'frame_label']
    csv_data = pd.DataFrame(columns=csv_colname, data=csv_list)
    csv_data.to_csv(csv_path)

    return 1


# examples: Call the above two functions
if __name__ == "__main__":
    src_path = r'F:\Xxy_Q\核查1\None/'
    label_path = r'F:\Xxy_Q\核查1\ImageNet_None/'
    img_save_path = r'F:\us_data\imageset_None/'
    # src_path = r'F:\ZDJ\our_data\zhujiang/'

    num_ind = 0
    for My_file_name in get_file_name(src_path):
        if My_file_name.find(".mp4") != -1:
            print(num_ind)
            num_ind += 1
            video_name = My_file_name.split('.')[0]
            video_id = My_file_name.split('_')[0]
            if not os.path.exists(img_save_path + video_id):
                os.mkdir(img_save_path + video_id)
            read_video_and_label(src_path + My_file_name, label_path, img_save_path + video_id + '/', video_name,
                                 video_id)
