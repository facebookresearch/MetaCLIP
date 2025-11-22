# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import tarfile

from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter


def tar_iter(fns):
    if isinstance(fns, str):
        fns = [fns]
    for fn in fns:
        with tarfile.open(fn) as tar:
            members = tar.getmembers()
            for member in members:
                if not member.name.endswith(".jpeg"):
                    continue # this should not happen
    
                uuid = member.name[:-len(".jpeg")] # refactor as get_uuid_from_file
                if uuid.startswith("./"):
                    uuid = img_uuid[len('./'):]
                yield uuid, member, tar


def transform_img_member(transform, member, tar):
    with tar.extractfile(member) as f:
        img = f.read()

    with Image.open(BytesIO(img)) as img:
        image = img.convert("RGB")
        image = transform(image)
    return image
