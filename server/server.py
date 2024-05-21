from clip_cpp import Clip
import os
from typing import Generator, Union
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from asyncio import Lock
import pickle
import mimetypes
from tqdm import tqdm

processable_types = {'image/png', 'image/jpeg'}
def is_processable_img(path:str):
    t, _ = mimetypes.guess_type(path)
    return t in processable_types and os.path.exists(path)

class MyClip:
    def __init__(self) -> None:
        # repo_id = 'mys/ggml_CLIP-ViT-B-32-laion2B-s34B-b79K'
        model_path = './models/CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q4_0.gguf'

        self.model = Clip(
            model_path_or_repo_id=model_path,
            # model_file=model_file,
            verbosity=2
        )

    def txt2emb(self, txt:str)->list[float]:
        "计算文字向量"
        tokens = self.model.tokenize(txt)
        emb = self.model.encode_text(tokens)
        return emb
    
    def img_path2emb(self, img_path:str)->list[float]:
        "计算一张图片的向量，使用路径"
        emb = self.model.load_preprocess_encode_image(img_path)
        return emb

    def calculate_similarity(self, emb0:list[float], emb1:list[float]):
        return self.model.calculate_similarity(emb0, emb1)

    def embs2similarity(self, emb:list[float], embs:Generator[list[float], None, None])->Generator[float, None, None]:
        "计算一对多相似度，生成器"
        for i in embs:
            yield self.calculate_similarity(emb, i)

    def img_paths2emb(self, img_paths:Union[list[str], set[str]])->Generator[list[float], None, None]:
        "计算一组图片的向量，生成器"
        for img_path in img_paths:
            yield self.img_path2emb(img_path)

# 使用fastapi来处理请求
app = FastAPI()
clip = MyClip()

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

# 全局图片路径-向量字典
if os.path.exists("./embeddings.pkl"):
    with open("./embeddings.pkl", "rb") as f:
        image_embeddings: dict[str, list[float]] = pickle.load(f)
else:
    image_embeddings: dict[str, list[float]] = {}
    with open("./embeddings.pkl", "wb") as f:
        pickle.dump(image_embeddings, f)
# 初始化一个锁
image_lock = Lock()

@app.post("/prep_imgs")
async def prep_imgs(img_paths: ImgPaths):
    "预处理图片，保存在字典里"
    async with image_lock:
        paths:set = set(img_paths.paths) - set(image_embeddings.keys())
        paths = set(filter(is_processable_img, paths))
        skip:int = len(img_paths.paths) - len(paths)
        generator = zip(paths, clip.img_paths2emb(paths))
        success:int = 0
        fail:int = 0

        for _ in tqdm(range(len(paths)), desc="Processing images", unit="img"):
            try:
                path, emb = next(generator)
                image_embeddings[path] = emb
                success += 1
            except Exception as e:
                fail += 1
                continue
            # 每五十张图片保存一次
            if success % 50 == 0:
                with open("./embeddings.pkl", "wb") as f:
                    pickle.dump(image_embeddings, f)

        with open("./embeddings.pkl", "wb") as f:
            pickle.dump(image_embeddings, f)

        return {"success": success, "fail": fail, "skip": skip}

def get_similarity_imgpaths(emb:list[float], lower_limit:float, top_n:int)->list[tuple[str, float]]:
    imgpaths_similarity = {}

    for target_imgpath, target_emb in image_embeddings.items():
        similarity = clip.calculate_similarity(emb, target_emb)
        if similarity > lower_limit:
            imgpaths_similarity[target_imgpath] = similarity

    imgpaths_similarity = list(imgpaths_similarity.items())
    imgpaths_similarity.sort(key=(lambda x:x[1]), reverse=True)

    return imgpaths_similarity[:top_n]

@app.post("/img2img")
async def img2img(img_lower_limit: ImgLowerLimit):
    "根据图片路径与提供的下限，返回对应的相似图片路径"
    if len(image_embeddings) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image embeddings dictionary is empty. Please ensure image preprocessing has been completed."
        )

    if is_processable_img(img_lower_limit.img_path):
        emb = clip.img_path2emb(img_lower_limit.img_path)
        imgpaths_similarity = get_similarity_imgpaths(emb, img_lower_limit.lower_limit, img_lower_limit.top_n)

        return imgpaths_similarity
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The provided image path is not processable."
        )


@app.post("/txt2img")
async def txt2img(txt_lower_limit: TxtLowerLimit):
    "根据文字与提供的下限，返回对应的相似图片路径"
    if len(image_embeddings) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image embeddings dictionary is empty. Please ensure image preprocessing has been completed."
        )

    emb = clip.txt2emb(txt_lower_limit.txt)
    imgpaths_similarity = get_similarity_imgpaths(emb, txt_lower_limit.lower_limit, txt_lower_limit.top_n)

    return imgpaths_similarity
