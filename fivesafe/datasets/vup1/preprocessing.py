import subprocess
import json
import os
from labelstudio_reader import generate_timedict_from_labelstudio_export

# TODO check if already done.
# TODO MAP TRACK NAMES
# TODO SKIP

class Preprocessor():
    """ Given videos of same fps, make images. """
    def __init__(self, top_view_cfg, perspective_view_cfg):
        self.top_view_cfg = top_view_cfg
        self.perspective_view_cfg = perspective_view_cfg

    def _chk_configs(self):
        # files exists,
        # output dirs empty
        return True

    @staticmethod
    def _convert_video_to_images(fname_video, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ffmpeg_img_slicing_from_video(fname_video, output_dir)
        print(f'converted video {fname_video} to images in {output_dir}.')

    def _map_track_names(self):
        pass

    def _convert_top_view_video(self):
        self._convert_video_to_images(
            self.top_view_cfg['input_dir'], 
            self.top_view_cfg['output_dir']
        )

    def _convert_perspective_view_videos(self):
        for _, cfg in self.perspective_view_cfg.items():
            self._convert_video_to_images(
                cfg['input_dir'], 
                cfg['output_dir']
            )

    def _get_width_height(self):
        pass

    def _create_label_files(self):
        # Perspective Views (TODO FCT)
        for _, cfg in self.perspective_view_cfg.items():
            timedict_pv = generate_timedict_from_labelstudio_export(cfg['labels'], 0)
            w, h = cfg['image_size_wh']
            timedict_pv = self._change_labels_from_percentage_to_absolute_values(timedict_pv, w, h)
            with open(os.path.join(cfg['output_dir'], 'labels.json'), 'w') as fp:
                json.dump(timedict_pv['timeseries'], fp)

        # Top View (TODO FCT)
        timedict_tv = generate_timedict_from_labelstudio_export(top_view_cfg['labels'], 1)
        w, h = top_view_cfg['image_size_wh']
        timedict_tv = self._change_labels_from_percentage_to_absolute_values(timedict_tv, w, h)
        with open(os.path.join(top_view_cfg['output_dir'], 'labels.json'), 'w') as fp:
            json.dump(timedict_tv['timeseries'], fp)

    @staticmethod
    def _change_labels_from_percentage_to_absolute_values(label_dict, width, height):
        for _, obj_dict in label_dict['timeseries'].items():
            if obj_dict != {}:
                for _, obj_values in obj_dict.items():
                    obj_values["x"] = obj_values["x"] * width/100
                    obj_values["y"] = obj_values["y"] * height/100
                    obj_values["width"] = obj_values["width"] * width/100
                    obj_values["height"] = obj_values["height"] * height/100
        return label_dict

    def preprocess(self):
        if not self._chk_configs():
            print('failed.')
            return 
        #self._convert_top_view_video()
        #self._convert_perspective_view_videos()
        self._create_label_files()


def ffmpeg_img_slicing_from_video(inputvideo, outputpath):
    command_line = f"ffmpeg -i {inputvideo}"
    print(command_line)
    pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
    output = pipe.read().decode()
    pipe.close()

    command_line = f"ffmpeg -i {inputvideo} {outputpath}/%05d.jpg"
    print(command_line)
    pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
    output = pipe.read().decode()
    pipe.close()


if __name__ == '__main__':
    top_view_cfg = {
        'input_dir': '/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/drone_ped_stab_h264.mp4',
        'output_dir': '/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/processed/top_view',
        'labels': '/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/cam2labels_and_topview_labels.json',
        'image_size_wh': [3840, 2160]
    }
    perspective_view_cfg = {
        1: {
            'input_dir': '/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/camera1_ped1_handbrake.mp4',
            'output_dir': '/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/processed/camera1',
            'offset_to_top_view': 40,
            'labels': '/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/cam1labels.json',
            'image_size_wh': [1920, 1002]
        },
        2: {
            'input_dir': '/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/camera2_ped1_handbrake.mp4',
            'output_dir': '/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/processed/camera2',
            'offset_to_top_view': 79,
            'labels': '/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/cam2labels_and_topview_labels.json',
            'image_size_wh': [1920, 1002]
        }
    }

    preprocessor = Preprocessor(top_view_cfg, perspective_view_cfg)
    preprocessor.preprocess()