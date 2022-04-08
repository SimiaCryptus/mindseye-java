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
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * This class tests the ReshapeLayer class.
 *
 * @docgenVersion 9
 */
public abstract class ReshapeLayerTest extends LayerTestBase {

  private final int[] outputDims;
  private final int[] inputDims;

  /**
   * Instantiates a new Reshape layer test.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  protected ReshapeLayerTest(int[] inputDims, int[] outputDims) {
    this.inputDims = inputDims;
    this.outputDims = outputDims;
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return new ReshapeLayer(outputDims);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{inputDims};
  }

  /**
   * The Basic class is a class that contains the most basic information.
   *
   * @docgenVersion 9
   */
  public static class Basic extends ReshapeLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(new int[]{6, 6, 1}, new int[]{1, 1, 36});
    }

  }

  /**
   * This class is called Basic1.
   *
   * @docgenVersion 9
   */
  public static class Basic1 extends ReshapeLayerTest {
    /**
     * Instantiates a new Basic 1.
     */
    public Basic1() {
      super(new int[]{1, 1, 32}, new int[]{1, 1, 32});
    }

  }

  /**
   * The Big0 class represents a big number.
   *
   * @author John Doe
   * @version 1.0, Januaray 1, 2000
   * @docgenVersion 9
   */
  public static class Big0 extends Big {
    /**
     * Instantiates a new Big 0.
     */
    public Big0() {
      super(256);
    }

  }

  /**
   * This class represents a big number.
   *
   * @docgenVersion 9
   */
  public static class Big1 extends Big {
    /**
     * Instantiates a new Big 1.
     */
    public Big1() {
      super(new int[]{4, 4, 256}, new int[]{1, 1, 2 * 2048});
    }
  }

  /**
   * The Big2 class is a representation of a card game.
   *
   * @author
   * @docgenVersion 9
   */
  public static class Big2 extends Big {
    /**
     * Instantiates a new Big 2.
     */
    public Big2() {
      super(new int[]{1, 1, 2 * 2048}, new int[]{4, 4, 256});
    }

  }

  /**
   * This class represents a big object.
   *
   * @docgenVersion 9
   */
  public abstract static class Big extends ReshapeLayerTest {

    /**
     * Instantiates a new Big.
     *
     * @param size the size
     */
    public Big(int size) {
      this(new int[]{1, 1, size}, new int[]{1, 1, size});
    }

    /**
     * Instantiates a new Big.
     *
     * @param inputDims  the input dims
     * @param outputDims the output dims
     */
    public Big(int[] inputDims, int[] outputDims) {
      super(inputDims, outputDims);
    }

    @Override
    public @Nullable BatchingTester getBatchingTester() {
      return getBatchingTester(1e-2, true, 5);
    }

    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

    @Override
    @Disabled
    public void derivativeTest() {
      super.derivativeTest();
    }

    @Override
    @Disabled
    public void jsonTest() {
      super.jsonTest();
    }

    @Override
    @Disabled
    public void perfTest() {
      super.perfTest();
    }

  }
}
