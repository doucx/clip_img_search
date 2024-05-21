import re
import shutil
from typing import Dict, Union
import requests
import os
import mimetypes
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler
import json
from pathlib import Path


def recursive_walk(directory: Path) -> list[Path]:
    file_paths = []

    for item in directory.iterdir():  # 使用iterdir遍历子目录和文件
        if item.is_dir():  # 检查是否为目录
            # 直接递归调用并extend结果，pathlib的Path对象可以直接操作，无需转换
            file_paths.extend(recursive_walk(item))
        else:
            # 直接添加到列表，Path对象即表示绝对路径
            file_paths.append(item)

    return file_paths


def multi_walk(directorys: list) -> list:
    file_paths = []
    for i in directorys:
        file_paths += recursive_walk(i)

    return file_paths


def get_names(paths: list) -> list:
    return list(map(os.path.basename, paths))


def get_usable_path(path: Path) -> Path:
    # 循环直到找到一个未被使用的文件名
    counter = 1
    while path.exists():
        path_split = path.stem.split("_")
        if len(path_split) > 1:
            try:
                # 构造新的文件夹名，通过检查是否已存在'_数字'后缀来决定如何拼接
                counter = int(path_split[-1])
                new_name = "_".join(path_split[:-1]) + f"_{counter+1}"
            except ValueError:
                # 否则，在原始名称后添加序号
                new_name = f"{path.stem}_{counter}"
            # 更新路径
        else:
            new_name = f"{path.stem}_{counter}"

        path = path.parent / (new_name + path.suffix)

    return path


processable_types = {'image/png', 'image/jpeg'}


def is_processable_img(path: Union[str, Path]) -> bool:
    t, _ = mimetypes.guess_type(path)
    return t in processable_types and os.path.exists(path)


# 加载配置
with open("./client_settings.yaml", "r") as file:
    settings = yaml.safe_load(file)

    imgs_path = Path(settings["imgs_path"]).absolute()
    img_search_path = Path(settings["img_search_path"]).absolute()
    txt_search_path = Path(settings["txt_search_path"]).absolute()

    server_url: str = settings["server_url"]

    img2imgs_point: str = settings["img2imgs_point"]
    txt2imgs_point: str = settings["txt2imgs_point"]
    prep_imgs_point: str = settings["prep_imgs_point"]


class Img2ImgsHandler(FileSystemEventHandler):
    def get_similarity(self, path: Path) -> list[tuple[str, float]]:
        "从服务器获取相似度"
        req = requests.post(server_url + img2imgs_point,
                            json={
                                "img_path": str(path.absolute()),
                            },
                            )
        print("Got", len(req.json()))
        similarity: list[tuple[str, float]] = req.json()
        return similarity

    def search(self, path: Path):
        """图搜图"""
        similarity = self.get_similarity(path)
        # 将得到的图片链接到文件夹里
        target_path = get_usable_path(
            img_search_path /
            ".".join(
                os.path.basename(path).split(".")[
                    :-
                    1]))
        target_path.mkdir()
        i = 0
        for img_path, s in similarity:
            os.link(img_path, target_path /
                    f"{i}_{s:.4f}_{os.path.basename(img_path)}")
            i += 1

    def on_created(self, event):
        path = Path(event.src_path)
        if path.parent.absolute() == img_search_path.absolute() and is_processable_img(path):
            print("img2imgs", path)
            self.search(path)


class Txt2ImgsHandler(FileSystemEventHandler):
    def get_similarity(self, path: Path) -> list[tuple[str, float]]:
        "从服务器获取相似度"
        with open(path, 'r') as f:
            txt = f.read()
        req = requests.post(server_url + txt2imgs_point,
                            json={
                                "txt": txt,
                            },
                            )
        print("Got", len(req.json()))
        similarity: list[tuple[str, float]] = req.json()
        return similarity

    def search(self, path: Path):
        """文搜图"""
        similarity = self.get_similarity(path)
        target_path = get_usable_path(
            txt_search_path /
            ".".join(
                path.name.split(".")[
                    :-
                    1]))
        target_path.mkdir()
        i = 0
        shutil.copy(path, target_path / path.name)
        for img_path, s in similarity:
            os.link(img_path, target_path /
                    f"{i}_{s:.4f}_{os.path.basename(img_path)}")
            i += 1

    def on_created(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        if path.is_file() and path.parent.absolute() == txt_search_path.absolute():
            print("txt2imgs", path)
            self.search(path)


class ImgCreateHandler(FileSystemEventHandler):
    """图片路径新增图片时，让服务器预处理该图片"""

    def prep_img(self, path):
        requests.post(server_url + prep_imgs_point,
                      json={
                          "paths": [str(path.absolute())]
                      })

    def on_created(self, event):
        path = Path(event.src_path)
        if is_processable_img(path):
            print("Processing created image", path)
            self.prep_img(path)


def prep_observer(event_handler, path):
    observer = Observer()

    # recursive=True 表示递归监控子目录
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    return observer


if __name__ == "__main__":
    img_paths: list = list(
        map(str, filter(is_processable_img, recursive_walk(imgs_path))))
    print("Processing images...")
    req = requests.post(server_url + prep_imgs_point, json={
        "paths": img_paths
    })
    print(req.json())

    observers = []
    observers.append(prep_observer(Img2ImgsHandler(), img_search_path))
    observers.append(prep_observer(Txt2ImgsHandler(), txt_search_path))
    observers.append(prep_observer(ImgCreateHandler(), imgs_path))

    try:
        while True:
            pass
    except KeyboardInterrupt:
        for o in observers:
            o.stop()
    for o in observers:
        o.join()
