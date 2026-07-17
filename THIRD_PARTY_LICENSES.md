# Third-Party Model Licenses

Macolorize bundles machine-learning models converted to Core ML format
(`.mlpackage`) from the following open-source projects. In each case the
original authors released both code and pretrained weights under the license
stated; the bundled files are derivative works (format conversion to Core ML
with added pre-/post-processing wrappers, precision conversion where noted).
Full license texts follow the summary table.

| Bundled model | Upstream project | Copyright | License | Modifications |
|---|---|---|---|---|
| `DDColor.mlpackage`, `DDColor_artistic.mlpackage` | [piddnad/DDColor](https://github.com/piddnad/DDColor) | © DDColor authors | Apache-2.0 | Traced, converted to Core ML mlprogram, FP16; fixed 512×512 input; ab-channel tensor output |
| `LaMa.mlpackage` | [advimman/lama](https://github.com/advimman/lama) (big-lama checkpoint) | © Samsung AI Center, LaMa authors | Apache-2.0 | Conversion approach adapted from [mallman/CoreMLaMa](https://github.com/mallman/CoreMLaMa) (Apache-2.0); fixed 800×800 input, FP32, mask binarization wrapper |
| `GFPGAN.mlpackage` | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) (v1.4, "clean" architecture) | © 2021 THL A29 Limited, a Tencent company | Apache-2.0 | Traced with deterministic noise buffers, converted to Core ML, FP32; [-1,1] domain wrapper |
| `RealESRGAN_x2plus.mlpackage`, `RealESRGAN_general_x4v3.mlpackage` | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | © 2021 Xintao Wang | BSD-3-Clause | Architectures re-declared for tracing, converted to Core ML, FP16; fixed 512×512 tile input |
| `DRUNet.mlpackage` | [cszn/KAIR](https://github.com/cszn/KAIR) (drunet_color) | © 2019 Kai Zhang | MIT | Converted to Core ML, FP16; scalar sigma → noise-map wrapper |
| `ScratchNet.mlpackage` | [microsoft/Bringing-Old-Photos-Back-to-Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) (scratch-detection U-Net; the README states "the codes and the pretrained model in this repository are under the MIT license") | © Microsoft Corporation | MIT | BatchNorm substituted for SyncBatchNorm (inference-equivalent), converted to Core ML, FP16; sigmoid output wrapper |

Changes to the upstream works are limited to what is listed above; no model
weights were retrained or fine-tuned.

---

## Apache License 2.0

Applies to: DDColor, LaMa (and the CoreMLaMa conversion approach), GFPGAN.

Full text: https://www.apache.org/licenses/LICENSE-2.0

> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use these files except in compliance with the License. Unless required by
> applicable law or agreed to in writing, software distributed under the
> License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
> OF ANY KIND, either express or implied. See the License for the specific
> language governing permissions and limitations under the License.

GFPGAN carries the additional notice: "Tencent is pleased to support the open
source community by making GFPGAN available. Copyright (C) 2021 THL A29
Limited, a Tencent company. All rights reserved."

## BSD 3-Clause License

Applies to: Real-ESRGAN. Copyright (c) 2021, Xintao Wang. All rights reserved.

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 3. Neither the name of the copyright holder nor the names of its
>    contributors may be used to endorse or promote products derived from this
>    software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.

## MIT License

Applies to: DRUNet/KAIR (Copyright (c) 2019 Kai Zhang) and the
Bringing-Old-Photos-Back-to-Life scratch-detection model (Copyright (c)
Microsoft Corporation).

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to
> deal in the Software without restriction, including without limitation the
> rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
> sell copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions: The above copyright
> notice and this permission notice shall be included in all copies or
> substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
> WITHOUT WARRANTY OF ANY KIND.

---

Sample and screenshot imagery: Library of Congress / Wikimedia Commons
public-domain photographs and synthetic test images only.
