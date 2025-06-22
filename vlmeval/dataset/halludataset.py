import os
import os.path as osp
import json
from PIL import Image
import numpy as np
import pandas as pd
import logging

from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.mlvu import *
from decord import VideoReader, cpu

FAIL_MSG = 'Failed to obtain answer via API.'

class Halludataset(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    def __init__(self, dataset='Halludataset',
                    #  json_file='/home/stud/xingyu/LMUData/hallucination/hallucinations_question.json',
                    frame_root='/home/stud/xingyu/LMUData/',
                    dataset_path = '/home/stud/xingyu/LMUData/',
                    nframe=0,
                    fps=-1,
                    pack=False):   
        # self.json_file = json_file
        self.frame_root = frame_root
        self.dataset_path = dataset_path

        super().__init__(dataset=dataset, nframe=nframe, fps=fps, pack=pack)

    @classmethod
    def supported_datasets(cls):
        return ['Halludataset']

    def prepare_dataset(self, dataset_name='Halludataset'):
        def check_integrity(pth):
            print(f"Path: {pth}")
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not os.path.exists(data_file):
                return False
            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['video'])):
                    print(f"Exists: {osp.exists(osp.join(pth, item['video']))}")
                    return False
            return True

        # try:
        #     with open(self.json_file, 'r') as f:
        #         ori_data = json.load(f)
        # except FileNotFoundError:
        #     logging.error(f"Annotation file not found: {self.json_file}")
        #     raise
        # except json.JSONDecodeError:
        #     logging.error(f"Error decoding JSON from: {self.json_file}")
        #     raise

        if self.dataset_path is not None and check_integrity(self.dataset_path):
            dataset_path = self.dataset_path
        else:
            self.data_list = []
            data_file = osp.join(self.dataset_path, f'{dataset_name}.tsv')
            for index in data_file:
                data = data_file[index]
                frames_dir = osp.join(self.frame_root, index)
                files = os.listdir(frames_dir)
                n_frames = len(files)
                for task_type in data:
                    questions = data[task_type]
                    if len(questions) > 0:
                        for question in questions:
                            self.data_list.append({
                                'video': index,
                                'task_type': task_type,
                                'question': question,
                                'options': ['Yes', 'No'],
                                'answer': 'No',
                                'n_frames': n_frames,
                            })
            data_df = pd.DataFrame(self.data_list[:10])
            data_df = data_df.assign(index=range(len(data_df)))
            data_df.to_csv(data_file, sep='\t', index=False)
            dataset_path = self.dataset_path

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def __len__(self):
        return len(self.videos) if self.pack else len(self.data)

    def __getitem__(self, idx):
        if self.pack:
            video = self.videos[idx]
            sub_data = [item for item in self.data if item['video'] == video]
            return sub_data
        else:
            return self.data[idx]

    def get_index_by_frame(self, max_frame):
        seg_size = float(max_frame) / self.num_segments
        frame_indices = np.array([
            int((seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def get_index_by_fps(self, vid, fps):
        total_frames = len(vid)
        video_fps = vid.get_avg_fps()
        total_duration = total_frames / video_fps
        required_frames = int(total_duration * fps)
        step_size = video_fps / fps
        frame_indices = np.array([int(i * step_size) for i in range(required_frames)])
        self.num_segments = len(frame_indices)
        return frame_indices

    def read_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1

        images_group = list()
        if self.fps < 0:
            frame_indices = self.get_index_by_frame(max_frame)
        else:
            frame_indices = self.get_index_by_fps(vr, self.fps)

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def save_video_frames(self, imgs, video_name, frames):
        if self.fps > 0:
            frame_paths = self.frame_paths_fps(video_name, frames)
        else:
            frame_paths = self.frame_paths(video_name)
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            block_size = imgs.size(0) // frames
            split_tensors = torch.split(imgs, block_size)
            to_pil = transforms.ToPILImage()
            images = [to_pil(arr) for arr in split_tensors]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    def save_video_into_images(self, line):
        video_path = os.path.join(self.data_root, line['prefix'], line['video'])
        if self.fps <= 0:
            self.num_segments = self.nframe
        else:
            self.num_segments = 0

        parent_dir = os.path.dirname(line['video']) # 添加

        torch_imgs = self.read_video(video_path)
        # img_frame_paths = self.save_video_frames(torch_imgs, line['video'], self.num_segments)
        img_frame_paths = self.save_video_frames(torch_imgs, parent_dir, self.num_segments)
        return img_frame_paths

    def build_prompt(self, idx, video_llm):
        if self.pack:
            raise NotImplementedError
        else:
            item = self.data.iloc[idx]
            return self._build_single_prompt(item, video_llm)


    def _build_single_prompt(self, item, video_llm):
        # question = f"Question: {item['question']}\n"
        # question += "Options:\n"
        # question += item['options'].replace('[','').replace(']','')
        # question = question.rstrip()

        question = f"Question: {item['question']}\n"
        question += 'Options:\n'

        option_keys = ['A', 'B', 'C', 'D']
        for key in option_keys:
            option_text = item[key]
            question += f"({key}) {option_text}\n"

        question = question.rstrip()


        message = [
            {"type": "text", "value": "Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question.", "role": "system"},
            {"type": "text", "value": question},
        ]

        video = item['video']

        # folder_path = osp.join(self.frame_root, video)
        # files = os.listdir(folder_path)
        # paths = [osp.join(folder_path, f) for f in files if f.lower().endswith('.jpg')]
        # paths.sort()
        # if self.nframe > 0:
        #     indices = np.linspace(0, len(paths) - 1, self.nframe).astype(int)
        #     frame_paths = [paths[i] for i in indices]
        # elif self.fps > 0:
        #     total_frames = item['n_frames']
        #     video_fps = 20
        #     total_duration = total_frames / video_fps
        #     required_frames = int(total_duration * self.fps)
        #     indices = np.linspace(0, len(paths) - 1, num=required_frames).astype(int)
        #     frame_paths = [paths[i] for i in indices]
        # else:
        #     frame_paths = paths

        # if video_llm:
        #     for frame_path in frame_paths:
        #         message.append(dict(type="image", value=frame_path))
        #     # message.append(dict(type="video", value=frame_paths))
        # else:
        #     for frame_path in frame_paths:
        #         message.append(dict(type="image", value=frame_path))
        video_path = os.path.join(self.data_root, item['video'])
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(item)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))

        message.append({"type": "text", "value": "\nOnly give the best option."})
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            model = judge_kwargs.setdefault('model', 'chatgpt-0125')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            # for idx in data['index']:
            #     ans = data.loc[data['index'] == idx, 'answer'].values[0]
            #     pred = data.loc[data['index'] == idx, 'prediction'].values[0]
            #     options = eval(data.loc[data['index'] == idx, 'options'].values[0])
            #     answer_idx = -1
            #     for id, c in enumerate(options):
            #         if c == ans:
            #             answer_idx = id
            #     ans = f"({chr(ord('A') + answer_idx)}) {ans}"
            #     input_item = data.loc[data['index'] == idx].to_dict(orient='records')[0]
            #     for id, option_content in enumerate(eval(input_item['options'])):
            #         input_item[chr(ord('A') + id)] = option_content
            #         if option_content == input_item['answer']:
            #             input_item['answer'] = chr(ord('A') + id)

            #     if FAIL_MSG in pred:
            #         data.loc[idx, 'score'] = -1
            #     else:
            #         data.loc[idx, 'score'] = int(check_ans_with_model(
            #             pred, ans, model,
            #             input_item,
            #             'Halludataset'
            #         ))

            # rejected = [x for x in data['score'] if x == -1]

            for idx in data['index']:
                row = data.loc[data['index'] == idx].to_dict(orient='records')[0]

                # 从 A/B/C/D 列构造 options 列表
                option_keys = ['A', 'B', 'C', 'D']
                options = [row[k] for k in option_keys]

                # 获取正确答案字母和内容
                answer_letter = row['answer']
                answer_idx = option_keys.index(answer_letter)
                answer_text = options[answer_idx]
                formatted_answer = f"({answer_letter}) {answer_text}"

                # 构造 input_item 传给 judge 模型
                input_item = row.copy()
                for i, opt in enumerate(options):
                    input_item[option_keys[i]] = opt
                input_item['answer'] = answer_letter  # 传的是 "A"/"B"/...
                input_item['task_type'] = 'default'


                pred = row['prediction']
                # if FAIL_MSG in pred:
                if FAIL_MSG in str(pred):
                    data.loc[data['index'] == idx, 'score'] = -1
                else:
                    data.loc[data['index'] == idx, 'score'] = int(check_ans_with_model(
                        pred, formatted_answer, model, input_item, 'Halludataset'
                    ))

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating