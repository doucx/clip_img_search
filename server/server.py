from contextlib import asynccontextmanager
from clip_cpp import Clip
from apscheduler.schedulers.background import BackgroundScheduler
import os
from typing import Union
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import pickle
import mimetypes
from tqdm.asyncio import tqdm
import numpy as np
import asyncio

processable_types = {'image/png', 'image/jpeg'}

class MyClip:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()

    def load_model(self):
        model_path = './models/CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q8_0.gguf'

        self.model = Clip(
            model_path_or_repo_id=model_path,
            verbosity=2
        )

    async def txt2emb(self, txt: str) -> list[float]:
        "计算文字向量"
        async with self.lock:
            tokens = self.model.tokenize(txt)
            emb = self.model.encode_text(tokens)
            return emb

    async def img_path2emb(self, img_path: str) -> list[float]:
        "计算一张图片的向量，使用路径"
        # TODO 使用自己的加载图片方法
        # TODO 解决内存泄漏

        async with self.lock:
            emb = self.model.load_preprocess_encode_image(img_path)
            return emb

    def cosine_similarity(self, vector_a, vector_b):
        """
        计算两个向量的余弦相似度。
        
        参数:
        vector_a, vector_b: 1-D numpy数组，表示两个向量。
        
        返回:
        两个向量的余弦相似度，范围在-1到1之间。
        """
        # 确保输入是numpy数组
        vector_a = np.array(vector_a)
        vector_b = np.array(vector_b)
        
        # 计算点积
        dot_product = np.dot(vector_a, vector_b)
        
        # 计算向量的模长（欧几里得范数）
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        
        # 避免除以零错误
        if norm_a == 0 or norm_b == 0:
            return 0  # 如果任一向量为零向量，则认为相似度为0
        
        # 计算并返回余弦相似度
        return dot_product / (norm_a * norm_b)

    def calculate_similarity(
            self, emb0: list[float], emb1: list[float]):
        "计算一对一相似度"
        return self.cosine_similarity(emb0, emb1)

def is_processable_img(path: str):
    t, _ = mimetypes.guess_type(path)
    return t in processable_types and os.path.exists(path)

class ImgPaths(BaseModel):
    "图片路径列表"
    paths: list[str]

class ImgLowerLimit(BaseModel):
    "图片路径与下限"
    img_path: str
    lower_limit: Union[float, None] = 0
    top_n: Union[int, None] = 50

class TxtLowerLimit(BaseModel):
    "文字与下限"
    txt: str
    lower_limit: Union[float, None] = 0
    top_n: Union[int, None] = 50

need_to_save = False
def save_embeddings():
    """保存image_embeddings到文件"""
    global need_to_save
    if need_to_save:
        with open("./embeddings.pkl", "wb") as f:
            pickle.dump(image_embeddings, f)
        print("Embeddings saved.")
        need_to_save = False


# 全局图片路径-向量字典
if os.path.exists("./embeddings.pkl"):
    with open("./embeddings.pkl", "rb") as f:
        image_embeddings: dict[str, list[float]] = pickle.load(f)
        image_embeddings = {k: v for k, v in image_embeddings.items() if os.path.exists(k)}
else:
    image_embeddings: dict[str, list[float]] = {}
    save_embeddings()

# 初始化调度器
scheduler = BackgroundScheduler(timezone="Asia/Shanghai")  # 请替换为你所在时区

# 添加任务：每20分钟执行一次save_embeddings
scheduler.add_job(save_embeddings, 'interval', minutes=20)

# 启动调度器
scheduler.start()

clip = MyClip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    "生命周期事件"
    # Load the ML model
    clip.load_model()
    yield
    # FastAPI应用关闭时，关闭调度器
    scheduler.shutdown()
    save_embeddings()

# fastapi部分
app = FastAPI(lifespan=lifespan)


@app.post("/prep_imgs")
async def prep_imgs(img_paths: ImgPaths):
    "预处理图片，保存在字典里"
    paths = set(img_paths.paths) - set(image_embeddings.keys())
    skip = len(img_paths.paths) - len(paths)
    success = 0
    fail = 0

    global need_to_save

    for path in tqdm(paths, desc="Processing images", unit="img"):
        try:
            if is_processable_img(path):
                emb = await clip.img_path2emb(path)
                image_embeddings[path] = emb
                success += 1
                need_to_save = True
                # 每五十张图片重置一次模型并保存
                if len(image_embeddings) % 50 == 0:
                    save_embeddings()
            else:
                fail += 1
        except Exception as e:
            fail += 1
            continue

    with open("./embeddings.pkl", "wb") as f:
        pickle.dump(image_embeddings, f)

    return {"success": success, "fail": fail, "skip": skip}


def get_similarity_imgpaths(
        emb: list[float], lower_limit: float, top_n: int) -> list[tuple[str, float]]:
    imgpaths_similarity = {}
    keys_to_remove = []

    for target_imgpath, target_emb in image_embeddings.items():
        if os.path.exists(target_imgpath):
            similarity = clip.calculate_similarity(emb, target_emb)
            if similarity > lower_limit:
                imgpaths_similarity[target_imgpath] = similarity
        else:
            keys_to_remove.append(target_imgpath)

    for k in keys_to_remove:
        del image_embeddings[k]

    imgpaths_similarity = list(imgpaths_similarity.items())
    imgpaths_similarity.sort(key=(lambda x: x[1]), reverse=True)

    return imgpaths_similarity[:top_n]


@app.post("/img2imgs")
async def img2imgs(img_lower_limit: ImgLowerLimit):
    "根据图片路径与提供的下限，返回对应的相似图片路径"
    print("img2imgs")
    if len(image_embeddings) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image embeddings dictionary is empty. Please ensure image preprocessing has been completed.")

    if is_processable_img(img_lower_limit.img_path):
        emb = await clip.img_path2emb(img_lower_limit.img_path)
        imgpaths_similarity = get_similarity_imgpaths(emb, img_lower_limit.lower_limit, img_lower_limit.top_n)

        return imgpaths_similarity
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The provided image path is not processable."
        )


@app.post("/txt2imgs")
async def txt2imgs(txt_lower_limit: TxtLowerLimit):
    "根据文字与提供的下限，返回对应的相似图片路径"
    print("txt2imgs")
    if len(image_embeddings) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image embeddings dictionary is empty. Please ensure image preprocessing has been completed.")

    emb = await clip.txt2emb(txt_lower_limit.txt)
    imgpaths_similarity = get_similarity_imgpaths(emb, txt_lower_limit.lower_limit, txt_lower_limit.top_n)

    return imgpaths_similarity
