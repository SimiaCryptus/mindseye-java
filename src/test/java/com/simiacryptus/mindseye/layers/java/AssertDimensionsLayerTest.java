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

import javax.annotation.Nonnull;

/**
 * The type Assert dimensions layer test.
 */
public abstract class AssertDimensionsLayerTest extends LayerTestBase {

  @Nonnull
  @Override
  public Layer getLayer() {
    return new AssertDimensionsLayer(2, 2);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{2, 2}};
  }

  /**
   * The type Basic.
   */
  public static class Basic extends AssertDimensionsLayerTest {

  }

}
