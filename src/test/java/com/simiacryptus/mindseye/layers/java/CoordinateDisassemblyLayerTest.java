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
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.mindseye.test.unit.BatchingTester;

import javax.annotation.Nonnull;

/**
 * The type Img band bias layer test.
 */
public abstract class CoordinateDisassemblyLayerTest extends LayerTestBase {

  public CoordinateDisassemblyLayerTest() {
    this.testingBatchSize = 1;
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return new CoordinateDisassemblyLayer();
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{2, 2, 3}};
  }

  @Override
  protected @Nonnull int[][] getLargeDims() {
    return new int[][]{{10, 10, 3}};
  }

  @Override
  public void batchingTest() {
    // Disable Test
  }

  @Override
  public void derivativeTest() {
    // Disable Test - Different size of TensorLists in input/output not supported.
    // Also, coordinate outputs not differentiated
  }

  /**
   * The type Basic.
   */
  public static class Basic extends CoordinateDisassemblyLayerTest {

  }

}