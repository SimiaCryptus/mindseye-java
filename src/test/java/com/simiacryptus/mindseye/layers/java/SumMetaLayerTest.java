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
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;

/**
 * The type Sum meta layer test.
 */
public abstract class SumMetaLayerTest extends MetaLayerTestBase {
  /**
   * Instantiates a new Sum meta layer test.
   */
  public SumMetaLayerTest() {
    super();
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{100}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    SumMetaLayer temp_73_0002 = new SumMetaLayer();
    temp_73_0002.setMinBatches(0);
    SumMetaLayer temp_73_0001 = temp_73_0002.addRef();
    temp_73_0002.freeRef();
    return temp_73_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{3}};
  }

  @Override
  @Disabled
  public void batchingTest() {
    super.batchingTest();
  }

  @Nonnull
  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends SumMetaLayerTest {

  }

}
