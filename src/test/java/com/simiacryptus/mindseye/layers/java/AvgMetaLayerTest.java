/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.MetaLayerTestBase;

import javax.annotation.Nonnull;

/**
 * The type Avg meta layer test.
 */
public abstract class AvgMetaLayerTest extends MetaLayerTestBase {

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{100}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    AvgMetaLayer temp_66_0002 = new AvgMetaLayer();
    temp_66_0002.setMinBatchCount(0);
    AvgMetaLayer temp_66_0001 = temp_66_0002.addRef();
    temp_66_0002.freeRef();
    return temp_66_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{3}};
  }

  @Nonnull
  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends AvgMetaLayerTest {

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new EntropyLossLayer();
    }
  }

}
