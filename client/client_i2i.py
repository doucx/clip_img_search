import re
from typing import Dict
import requests
import os
import mimetypes
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler
import json
from pathlib import Path


def recursive_walk(directory:str)->list:
    file_paths = []
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            # 递归调用时传入绝对路径，确保后续处理都是基于绝对路径的
            file_paths.extend(recursive_walk(item_path))
        else:
            # 添加绝对路径到file_paths列表
            file_paths.append(os.path.abspath(item_path))

    return file_paths

def multi_walk(directorys:list)->list:
    file_paths = []
    for i in directorys:
        file_paths += recursive_walk(i)
    
    return file_paths


def get_names(paths:list)->list:
    return list(map(os.path.basename, paths))

processable_types = {'image/png', 'image/jpeg'}
def is_processable_img(path):
    t, _ = mimetypes.guess_type(path)
    return t in processable_types and os.path.exists(path)

# 加载配置
with open("./client_settings.yaml", "r") as file:
    settings = yaml.safe_load(file)
    imgs_path = Path(settings["img_path"]).absolute()
    img_search_path = Path(settings["img_search_path"]).absolute()
    txt_search_path = Path(settings["txt_search_path"]).absolute()

class Img2ImgsHandler(FileSystemEventHandler):
    def get_similarity(self, path: Path) -> list[tuple[str, float]]:
        "从服务器获取相似度"
        req = requests.post(settings["img2img_url"],
                        json={
                          "img_path": str(path.absolute()),
                        },
                    )
        print("Got", len(req.json()))
        similarity:list[tuple[str, float]] = req.json()
        return similarity
    
    def search(self, path:Path):
        """图搜图"""
        similarity = self.get_similarity(path)
        # 将得到的图片链接到文件夹里
        target_path = img_search_path / ".".join(os.path.basename(path).split(".")[:-1])
        if target_path.exists():
            target_path.rmdir()
        target_path.mkdir()
        i = 0
        for img_path, s in similarity:
            os.link(imgs_path, target_path / f"{i}_{s:.4f}_{os.path.basename(img_path)}")
            i += 1

    def on_created(self, event):
        print("img2imgs", event.src_path)
        if Path(event.src_path).parent.absolute() == img_search_path.absolute() and is_processable_img(event.src_path):
            self.search(Path(event.src_path))

class Txt2ImgsHandler(FileSystemEventHandler):
    def get_similarity(self, path: Path) -> list[tuple[str, float]]:
        "从服务器获取相似度"
        with open(path, 'r') as f:
            txt = f.read()
        req = requests.post(settings["txt2img_url"],
                        json={
                          "txt": txt,
                        },
                    )
        print("Got", len(req.json()))
        similarity:list[tuple[str, float]] = req.json()
        return similarity
    
    def search(self, path:Path):
        """文搜图"""
        similarity = self.get_similarity(path)
        target_path = txt_search_path / ".".join(os.path.basename(path).split(".")[:-1])
        if target_path.exists():
            target_path.rmdir()
        target_path.mkdir()
        i = 0
        for img_path, s in similarity:
            os.link(img_path, target_path / f"{i}_{s:.4f}_{os.path.basename(img_path)}")
            i += 1

    def on_modified(self, event: FileSystemEvent) -> None:
        print("txt2imgs", event.src_path)
        if Path(event.src_path).parent.absolute() == txt_search_path.absolute():
            self.search(Path(event.src_path))

class ImgCreateHandler(FileSystemEventHandler):
    """在目标路径创建图片时，让服务器预处理该图片"""
    def prep_img(self, path):
        requests.post(settings['prep_imgs_url'], json={
                "paths": [path]
            })

    def on_created(self, event):
        if is_processable_img(event.src_path):
            print("Processing created image", event.src_path)
            self.prep_img(event.src_path)

def prep_observer(event_handler, path):
    observer = Observer()
    
    observer.schedule(event_handler, path, recursive=True)  # recursive=True 表示递归监控子目录
    observer.start()
    return observer

if __name__ == "__main__":
    images:list = list(filter(is_processable_img, recursive_walk(settings["img_path"])))
    print("Processing images...")
    req = requests.post(settings['prep_imgs_url'], json={
            "paths": images
        })
    print(req.json())

    img2img_event_handler = Img2ImgsHandler()
    img2img_observer = Observer()
    
    img2img_observer.schedule(img2img_event_handler, settings["img_search_path"], recursive=True)  # recursive=True 表示递归监控子目录
    img2img_observer.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        img2img_observer.stop()
    img2img_observer.join()
