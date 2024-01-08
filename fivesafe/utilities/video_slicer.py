import subprocess

class VideoSlicer():
    def __init__(self):
        pass

    def slice(self, input_file, output_dir):
        command_line = f"ffmpeg -i {input_file}"
        print(command_line)
        pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
        output = pipe.read().decode()
        pipe.close()

        command_line = f"ffmpeg -i {input_file} {output_dir}/%05d.jpg"
        print(command_line)
        pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
        output = pipe.read().decode()
        pipe.close()


if __name__ == '__main__':
    vs = VideoSlicer()
    vs.slice(
        input_file = '/home/ziegleto/ziegleto/data/5Safe/peter_paul/petpa/data/camera_1/handbraked/1/1.mp4',
        output_dir = '/home/ziegleto/ziegleto/data/5Safe/peter_paul/petpa/data/camera_1/handbraked/1/imgs'
    )
    print('done')