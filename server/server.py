from contextlib import asynccontextmanager
from clip_cpp import Clip
from apscheduler.schedulers.background import BackgroundScheduler
import os
from typing import AsyncGenerator, Generator, Union
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from asyncio import Lock
import asyncio
import pickle
import mimetypes
import uvicorn
from tqdm.asyncio import tqdm

processable_types = {'image/png', 'image/jpeg'}
def is_processable_img(path:str):
    t, _ = mimetypes.guess_type(path)
    return t in processable_types and os.path.exists(path)

class MyClip:
    def __init__(self) -> None:
        self.lock = Lock()

    def load_model(self):
        model_path = './models/CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q4_0.gguf'

        self.model = Clip(
            model_path_or_repo_id=model_path,
            verbosity=2
        )

    def _txt2emb_internal(self, txt:str)->list[float]:
        "计算文字向量"
        tokens = self.model.tokenize(txt)
        emb = self.model.encode_text(tokens)
        return emb

    async def txt2emb(self, txt:str)->list[float]:
        "计算文字向量"
        async with self.lock:
            return await asyncio.to_thread(self._txt2emb_internal, txt)
    
    def _img_path2emb_internal(self, img_path:str)->list[float]:
        "计算一张图片的向量，使用路径"
        emb = self.model.load_preprocess_encode_image(img_path)
        return emb

    async def img_path2emb(self, img_path:str)->list[float]:
        "计算一张图片的向量，使用路径"
        async with self.lock:
            return await asyncio.to_thread(self._img_path2emb_internal, img_path)

    def _calculate_similarity_internal(self, emb0:list[float], emb1:list[float]):
        "计算一对一相似度"
        return self.model.calculate_similarity(emb0, emb1)

    async def calculate_similarity(self, emb0:list[float], emb1:list[float]):
        "计算一对一相似度"
        async with self.lock:
            return await asyncio.to_thread(self._calculate_similarity_internal, emb0, emb1)

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


def save_embeddings():
    """保存image_embeddings到文件"""
    with open("./embeddings.pkl", "wb") as f:
        pickle.dump(image_embeddings, f)
    print("Embeddings saved.")

# 全局图片路径-向量字典
if os.path.exists("./embeddings.pkl"):
    with open("./embeddings.pkl", "rb") as f:
        image_embeddings: dict[str, list[float]] = pickle.load(f)
else:
    image_embeddings: dict[str, list[float]] = {}
    save_embeddings()

# 初始化调度器
scheduler = BackgroundScheduler(timezone="Asia/Shanghai")  # 请替换为你所在时区

# 添加任务：每五分钟执行一次save_embeddings
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

## fastapi部分
app = FastAPI(lifespan=lifespan)

@app.post("/prep_imgs")
async def prep_imgs(img_paths: ImgPaths):
    "预处理图片，保存在字典里"
    paths:set = set(img_paths.paths) - set(image_embeddings.keys())
    paths = set(filter(is_processable_img, paths))
    skip:int = len(img_paths.paths) - len(paths)
    success:int = 0
    fail:int = 0

    for path in tqdm(paths, desc="Processing images", unit="img"):
        try:
            emb = await clip.img_path2emb(path)
            image_embeddings[path] = emb
            success += 1
            # 每五十张图片重置一次模型并保存
            if len(image_embeddings) % 50 == 0:
                save_embeddings()
                # del clip.model
                # import gc; gc.collect()
                # clip.load_model()
                # print("reloaded model")
        except Exception as e:
            fail += 1
            continue

    with open("./embeddings.pkl", "wb") as f:
        pickle.dump(image_embeddings, f)

    return {"success": success, "fail": fail, "skip": skip}

async def get_similarity_imgpaths(emb:list[float], lower_limit:float, top_n:int)->list[tuple[str, float]]:
    imgpaths_similarity = {}

    for target_imgpath, target_emb in image_embeddings.items():
        similarity = await clip.calculate_similarity(emb, target_emb)
        if 1>= similarity > lower_limit:
            imgpaths_similarity[target_imgpath] = similarity

    imgpaths_similarity = list(imgpaths_similarity.items())
    imgpaths_similarity.sort(key=(lambda x:x[1]), reverse=True)

    return imgpaths_similarity[:top_n]

@app.post("/img2imgs")
async def img2imgs(img_lower_limit: ImgLowerLimit):
    "根据图片路径与提供的下限，返回对应的相似图片路径"
    if len(image_embeddings) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image embeddings dictionary is empty. Please ensure image preprocessing has been completed."
        )

    if is_processable_img(img_lower_limit.img_path):
        emb = await clip.img_path2emb(img_lower_limit.img_path)
        imgpaths_similarity = await get_similarity_imgpaths(emb, img_lower_limit.lower_limit, img_lower_limit.top_n)

        return imgpaths_similarity
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The provided image path is not processable."
        )


@app.post("/txt2imgs")
async def txt2imgs(txt_lower_limit: TxtLowerLimit):
    "根据文字与提供的下限，返回对应的相似图片路径"
    if len(image_embeddings) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image embeddings dictionary is empty. Please ensure image preprocessing has been completed."
        )

    emb = await clip.txt2emb(txt_lower_limit.txt)
    imgpaths_similarity = await get_similarity_imgpaths(emb, txt_lower_limit.lower_limit, txt_lower_limit.top_n)

    return imgpaths_similarity
