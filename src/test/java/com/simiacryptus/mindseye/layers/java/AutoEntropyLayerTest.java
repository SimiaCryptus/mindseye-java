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
 * This class tests the AutoEntropyLayer class.
 *
 * @docgenVersion 9
 */
public abstract class AutoEntropyLayerTest extends MetaLayerTestBase {

  /**
   * Instantiates a new Auto entropy layer test.
   */
  public AutoEntropyLayerTest() {
    super();
    //validateDifferentials = false;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{100}};
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return new AutoEntropyLayer();
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{3}};
  }

  /**
   * The Basic class is a class that contains the most basic information.
   *
   * @docgenVersion 9
   */
  public static class Basic extends AutoEntropyLayerTest {

    @Nonnull
    @Override
    protected Layer lossLayer() {
      return new EntropyLossLayer();
    }
  }

}
