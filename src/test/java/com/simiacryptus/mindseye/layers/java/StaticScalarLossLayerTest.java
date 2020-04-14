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
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;

/**
 * The type Static scalar loss layer test.
 */
public abstract class StaticScalarLossLayerTest extends LayerTestBase {

  /**
   * Instantiates a new Static scalar loss layer test.
   */
  public StaticScalarLossLayerTest() {
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return new StaticScalarLossLayer();
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{1}};
  }

  @Override
  @Disabled
  public void batchingTest() {
    super.batchingTest();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends StaticScalarLossLayerTest {

  }

}
