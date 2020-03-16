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

public abstract class ReshapeLayerTest extends LayerTestBase {

  private final int[] outputDims;
  private final int[] inputDims;

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

  public static class Basic extends ReshapeLayerTest {
    public Basic() {
      super(new int[]{6, 6, 1}, new int[]{1, 1, 36});
    }

  }

  public static class Basic1 extends ReshapeLayerTest {
    public Basic1() {
      super(new int[]{1, 1, 32}, new int[]{1, 1, 32});
    }

  }

  public static class Big0 extends Big {
    public Big0() {
      super(256);
    }

  }

  public static class Big1 extends Big {
    public Big1() {
      super(new int[]{4, 4, 256}, new int[]{1, 1, 2 * 2048});
    }
  }

  public static class Big2 extends Big {
    public Big2() {
      super(new int[]{1, 1, 2 * 2048}, new int[]{4, 4, 256});
    }

  }

  public abstract static class Big extends ReshapeLayerTest {

    public Big(int size) {
      this(new int[]{1, 1, size}, new int[]{1, 1, size});
    }

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
