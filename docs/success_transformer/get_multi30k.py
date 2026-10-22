import gzip
import os
import urllib.request

base = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
files = {
    "train": "train",
    "val": "val",
    "test_2016_flickr": "test2016",
}  # 원본 이름 → legacy가 찾는 이름
os.makedirs(".data/multi30k", exist_ok=True)
for src, dst in files.items():
    for ext in (".en", ".de"):
        with urllib.request.urlopen(base + src + ext + ".gz") as r:
            data = gzip.decompress(r.read())
        open(f".data/multi30k/{dst}{ext}", "wb").write(data)
        print("ok", dst + ext)
