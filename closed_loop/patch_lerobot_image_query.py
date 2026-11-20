#!/usr/bin/env python
"""Speed fix for LeRobot 0.4.4 when a dataset stores frames as images instead of video.

To fetch the next 50 actions for one sample, `LeRobotDataset._query_hf_dataset` indexes whole rows,
which also decodes the 50 images in those rows. A sample then takes over a second and training is
bound by the data loader. This patch makes the query read only the column it needs. It changes no
values: the same actions come back.

Usage: python patch_lerobot_image_query.py          (edits the installed file once; keeps a .orig copy)
"""
import os
import shutil

import lerobot.datasets.lerobot_dataset as mod

OLD = """            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
"""
NEW = """            views = self.__dict__.setdefault("_single_column_views", {})
            if views.get("_source") is not self.hf_dataset:  # the table was (re)loaded
                views.clear()
                views["_source"] = self.hf_dataset
            if key not in views:
                views[key] = self.hf_dataset.select_columns([key])
            result[key] = torch.stack(views[key][relative_indices][key])
"""


def main():
    path = mod.__file__
    src = open(path).read()
    if NEW in src:
        print("already patched:", path)
        return
    if OLD not in src:
        raise SystemExit("the expected code was not found; this patch is written for LeRobot 0.4.4")
    if not os.path.exists(path + ".orig"):
        shutil.copy(path, path + ".orig")
    open(path, "w").write(src.replace(OLD, NEW))
    print("patched:", path)


if __name__ == "__main__":
    main()
