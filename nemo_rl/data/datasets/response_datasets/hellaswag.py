# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class HellaSwagDataset(RawDataset):
    """HellaSwag commonsense natural-language inference.

    Each example gives a context and four candidate endings, one of which is correct.
    HellaSwag is normally an evaluation benchmark, so turning it into supervised
    fine-tuning data requires choosing what the model is asked to produce. The two
    reasonable readings differ by 2.4x in tokens per example, which matters for
    throughput work, so both are available:

        choice      (default) context plus all four endings, target is the correct
                    ending. Matches how HellaSwag is posed as a task.
                    Mean 118 tokens with the Nemotron-V3 tokenizer.
        completion  context only, target is the correct ending. The plain
                    language-modelling reading. Mean 48 tokens.

    For reference, SQuAD under the same tokenizer averages 214 tokens, so HellaSwag is
    the shorter workload either way.

    Args:
        split: dataset split, default "train" (39,905 examples)
        style: "choice" or "completion", see above
        subset: HuggingFace config name; HellaSwag has no configs, accepted and ignored
            so that a YAML passing ``subset: null`` does not warn
    """

    def __init__(
        self,
        split: str = "train",
        style: str = "choice",
        subset: str | None = None,
        **kwargs,
    ) -> None:
        if style not in ("choice", "completion"):
            raise ValueError(
                f"Unknown style {style!r} for HellaSwagDataset; expected 'choice' or 'completion'."
            )
        self.task_name = "hellaswag"
        self.style = style

        ds = load_dataset("Rowan/hellaswag", subset)[split]
        # The test split ships with empty labels, so there is no target to train on.
        ds = ds.filter(lambda r: r["label"] != "")

        self.dataset = ds.map(self.format_data, remove_columns=ds.column_names)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        correct = data["endings"][int(data["label"])]
        if self.style == "choice":
            options = "\n".join(
                f"{i + 1}. {ending}" for i, ending in enumerate(data["endings"])
            )
            user = f"{data['ctx']}\n\nWhich ending is correct?\n{options}"
        else:
            user = data["ctx"]

        return {
            "messages": [
                {"role": "system", "content": data["activity_label"]},
                {"role": "user", "content": user},
                {"role": "assistant", "content": correct},
            ],
            "task_name": self.task_name,
        }
