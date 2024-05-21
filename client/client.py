import re
from typing import Dict
import requests
import os
import mimetypes
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler
import json


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
def is_processable_img(path:str):
    t, _ = mimetypes.guess_type(path)
    return t in processable_types and os.path.exists(path)

# 加载配置
with open("./client_settings.yaml", "r") as file:
    settings = yaml.safe_load(file)

class Img2ImgsHandler(FileSystemEventHandler):
    def get_similarity(self, path: str, top_n: int, lower_limit: float) -> Dict[str, float]:
        similarity:Dict[str, float] = requests.post(settings["img2img_url"],
                        json={
                          "img": path,
                          "top_n": top_n,
                          "lower_limit": lower_limit
                        },
                        timeout=10
                    )
        return similarity
    
    def search(self, path:str):
        """图搜图"""
        if is_processable_img(path):
            try:
                basename = os.path.basename(path)
                top_n, lower_limit = basename.split(",")
                top_n = int(top_n)
                lower_limit = float(lower_limit)

                similarity = self.get_similarity(path, top_n, lower_limit)
            except Exception as e:
                print("fail to i2i")
                return

            os.mkdir(settings["img_search_path"]+os.path.basename(path).split(".")[0])
            i = 0
            for img_path, s in similarity.items():
                os.link(img_path, settings["img_search_path"] + f"{i}_{s:.4f}_{os.path.basename(img_path)}")
                i += 1

    # def on_created(self, event):
    #     self.search(event.src_path)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        self.search(event.src_path)

class Txt2ImgsHandler(FileSystemEventHandler):
    def get_similarity(self, path: str) -> Dict[str, float]:
        with open(path, "r") as f:
            j = json.load(f)
        similarity:Dict[str, float] = requests.post(settings["txt2img_url"],
                        json=j,
                        timeout=10
                    )
        return similarity
    
    def search(self, path:str):
        """文搜图"""
        if is_processable_img(path):
            try:
                similarity = self.get_similarity(path)
            except Exception as e:
                print("fail to t2i")
                return

            print("t2i")
            os.mkdir(settings["img_search_path"]+os.path.basename(path).split(".")[0])
            i = 0
            for img_path, s in similarity.items():
                os.link(img_path, settings["img_search_path"] + f"{i}_{s:.4f}_{os.path.basename(img_path)}")
                i += 1

    # def on_created(self, event):
    #     self.search(event.src_path)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        breakpoint()
        print("m")
        self.search(event.src_path)

class ImgCreateHandler(FileSystemEventHandler):
    def prep_img(self, path):
        if is_processable_img(path):
            print("Processing created image")
            requests.post(settings['prep_imgs_url'], json={
                    "paths": [path]
                })


    def on_created(self, event):
        self.prep_img(event.src_path)

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

    txt2img_event_handler = Txt2ImgsHandler()
    txt2img_observer = Observer()
    
    txt2img_observer.schedule(txt2img_event_handler, settings["txt_search_path"], recursive=True)  # recursive=True 表示递归监控子目录
    txt2img_observer.start()

    img_create_event_handler = ImgCreateHandler()
    img_create_observer = Observer()
    
    img_create_observer.schedule(img_create_event_handler, settings["img_path"], recursive=True)  # recursive=True 表示递归监控子目录
    img_create_observer.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        img2img_observer.stop()
        txt2img_observer.stop()
        img_create_observer.stop()
    img2img_observer.join()
    txt2img_observer.join()
    img_create_observer.join()
