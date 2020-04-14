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
 * The type Bias layer test.
 */
public abstract class BiasLayerTest extends LayerTestBase {

  private final int dimension;

  /**
   * Instantiates a new Bias layer test.
   *
   * @param dimension the dimension
   */
  public BiasLayerTest(int dimension) {
    this.dimension = dimension;
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    BiasLayer temp_75_0002 = new BiasLayer(dimension);
    temp_75_0002.addWeights(this::random);
    BiasLayer temp_75_0001 = temp_75_0002.addRef();
    temp_75_0002.freeRef();
    return temp_75_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{dimension}};
  }

  /**
   * The type Basic.
   */
  public static class Basic extends BiasLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(5);
    }

  }

  /**
   * The type Reducing.
   */
  public static class Reducing extends BiasLayerTest {

    /**
     * Instantiates a new Reducing.
     */
    public Reducing() {
      super(5);
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      BiasLayer temp_75_0004 = new BiasLayer(1);
      temp_75_0004.addWeights(this::random);
      BiasLayer temp_75_0003 = temp_75_0004.addRef();
      temp_75_0004.freeRef();
      return temp_75_0003;
    }

  }

}
