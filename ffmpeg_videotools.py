import subprocess
import glob
import os


def ffmpeg_video_creation(img_dir_path, outpath):
    duration = 0.05
    filenames = list()
    print(os.getcwd())
    print(img_dir_path)
    os.chdir(img_dir_path)
    for file in glob.glob("*.jpg"):
        filenames.append(file)

    with open("ffmpeg_input.txt", "wb") as outfile:
        for filename in filenames:
            outfile.write(f"file '{img_dir_path}/{filename}'\n".encode())
            outfile.write(f"duration {duration}\n".encode())

    command_line = f"ffmpeg -r 29.97 -f concat -safe 0 -i ffmpeg_input.txt -c:v libx264 -pix_fmt yuv420p {outpath}out.mp4"
    print(command_line)

    pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
    output = pipe.read().decode()
    pipe.close()

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


# 0. Image Stab (maybe not in pipeline)
# 1. Handbrake to make every vid 29.97 fps (like drone) -> maybe with python or handbrake in cmd-line
# 2. Bilder draus machen / Theoretisch geht auch video, wenn das geht... cv2
# 3. Mapping zwischen kameras
# 4. labels einlesen, feddich.
# 5. Dataset class..

if __name__ == "__main__":
    # TODO STABILISIERUNG AUCH REIN
    # TODO MUST DO THIS PIPELINE FOR EVERY VIDEO? CONFIG. TOP VIEW, PERSPECTIVE VIEWS.
    # TODO 1. FPS 29.97
    inputvideo = "/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/camera1_ped1_handbrake.mp4"
    outputpath_imgs = "/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/processed/camera1/imgs"
    outputpath_vid = "/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/processed/camera1"

    # Slice Video: make imgs out of video
    ffmpeg_img_slicing_from_video(inputvideo, outputpath_imgs)
    # Make Video: make fps 29.97
    #ffmpeg_video_creation(outputpath_imgs, outputpath_vid)