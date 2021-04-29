# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

df1 = pd.read_csv('../datasets/winequality-white.csv')
df2 = pd.read_csv('../datasets/winequality-red.csv')

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv('../datasets/winequality-full.csv')
