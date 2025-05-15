import os
import json
import base64
from pathlib import Path
# from PIL import Image
# import openai
# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector
import cv2
from tqdm import tqdm
import pandas as pd
import subprocess


def detect_scenes_cli(video_path, output_dir, threshold=20):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    # output_dir = video_path.parent
    base_name = video_path.stem
    csv_path = output_dir / f"{base_name}-Scenes.csv"

    subprocess.run([
        "scenedetect",
        "-i", str(video_path),
        "detect-content",
        f"--threshold={threshold}",
        "list-scenes",
        "-o", str(output_dir)
    ], check=True)

    return csv_path


def read_start_frames_from_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=1)
    return df['Start Frame'].astype(int).tolist()


def extract_frames_with_cv(video_path, frame_numbers, output_dir, grid_size=(3, 3)):
    Path(output_dir).mkdir(exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    grid_w, grid_h = grid_size
    num_required = grid_w * grid_h

    for i, start_frame in tqdm(enumerate(frame_numbers), total=len(frame_numbers), desc="抽取关键帧"):
        end_frame = frame_numbers[i + 1] if i < len(frame_numbers) - 1 else total_frames - 1

        available_range = end_frame - start_frame
        if available_range <= num_required + 2:
            print(f"⚠️ 场景 {i} 太短（帧数不足 {num_required + 2}），跳过")
            continue

        # 从 start+1 到 end-1 中抽取 num_required 个等间隔帧
        step = (end_frame - start_frame - 2) / (num_required - 1)
        selected_frames = [int(start_frame + 1 + round(j * step)) for j in range(num_required)]

        frames = []
        for frame_num in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ 无法读取帧 {frame_num}，跳过场景 {i}")
                frames = []
                break
            frames.append(frame)

        if len(frames) != num_required:
            print(f"⚠️ 场景 {i} 抽帧失败，不完整，跳过")
            continue

        # 拼成 grid_w * grid_h 图像
        try:
            rows = [cv2.hconcat(frames[y * grid_w:(y + 1) * grid_w]) for y in range(grid_h)]
            grid_image = cv2.vconcat(rows)
            output_img = f"{output_dir}/scene_{i:03d}.jpg"
            cv2.imwrite(output_img, grid_image)
        except Exception as e:
            print(f"❌ 拼图失败 场景 {i}：{e}")

    cap.release()


def generate_json_template(frame_numbers, output_dir, output_path):
    output = []
    for i, frame in enumerate(frame_numbers):
        output.append({
            "start_frame": frame,
            "image": f"scene_{i:03d}.jpg",
            "description": "请描述这张图代表的场景"
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


def describe_image(
    image_path,
    client,
    model: str = "default_model",
    sys_info: str = (
        "你是一个擅长视频场景理解的 AI，请根据提供的拼图图像，生成统一格式的视频场景描述。\n"
        "请包括：\n"
        "1. 场景中发生的主要事件或动作。\n"
        "2. 出现的人物（如有）及其行为。\n"
        "3. 场景的背景和气氛。\n"
        "请用简洁、客观、第三人称的方式描述，不要加入主观感受。\n"
        "输出格式如下：\n"
        "场景描述：XXX。"
    )
):
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
            b64_img = base64.b64encode(img_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_info},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请根据图像生成一段客观的视频场景描述，内容包括人物、动作、背景、氛围。输出请以“场景描述：”开头。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 处理图片 {image_path} 时出错: {e}")
        return "描述生成失败"


def fill_descriptions(
    client,
    input_json,
    output_json,
    output_dir,
    model
):
    with open(input_json, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    for scene in tqdm(scenes, desc="生成场景描述"):
        img_path = os.path.join(output_dir, scene["image"])
        scene["description"] = describe_image(img_path, client=client, model=model)
        scene.pop("image", None)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=4, ensure_ascii=False)

    print(f"✅ 完成：{output_json}")


class SceneDetector(object):
    def __init__(
        self,
        client,
        video_file,
        pre_processing: bool = False,
        temp_json: str = None,
        final_json: str = None
    ):
        self.client = client
        self.video_file = video_file
        self.pre_processsing = pre_processing
        self.temp_json = temp_json
        self.final_json = final_json

        if not self.temp_json:
            self.temp_json = self.video_file + ".tmp.json"
        if not self.final_json:
            self.final_json = self.video_file + ".final.json"

        if self.pre_processsing:
            self.pre_process()

    def pre_process(self):
        pass

    def detect(
        self,
        out_dir,
        model,
        grid_size=(3, 3)
    ):
        output_csv = detect_scenes_cli(self.video_file, out_dir)
        # df = pd.read_csv(output_csv, skiprows=1)
        # df = read_start_frames_from_csv(output_csv)
        starts = read_start_frames_from_csv(output_csv)
        extract_frames_with_cv(self.video_file, starts, out_dir, grid_size=grid_size)
        generate_json_template(starts, out_dir, self.temp_json)
        fill_descriptions(
            self.client,
            self.temp_json,
            self.final_json,
            out_dir,
            model=model
        )

