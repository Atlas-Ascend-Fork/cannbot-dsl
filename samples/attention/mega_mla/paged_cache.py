# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Resolve request-local cache pages at the point of a device cache read."""

from cannbotdsl import dtypes
from cannbotdsl.lang.jit import jit


class PagedCacheAddressing:
    """T=B*S tokens use table row tok//S; stride zero selects physical indices."""

    def __init__(self, page_size, batch_stride=0):
        self.page_size = page_size
        self.batch_stride = batch_stride

    @jit
    def page(self, index, block_table_gm, tok):
        # 负索引和有效前缀外的索引由调用方先屏蔽，不访问页表或cache。
        page = dtypes.int64(index // self.page_size)
        if self.batch_stride:
            page = dtypes.int64(block_table_gm[tok // self.batch_stride, page])
        return page
