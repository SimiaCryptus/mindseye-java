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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware class FullyConnectedLayerTest extends LayerTestBase {

  @Nonnull
  private final FullyConnectedLayer fullyConnectedLayer;
  private final int inputs;
  private final int outputs;

  protected FullyConnectedLayerTest(int inputs, int outputs) {
    fullyConnectedLayer = new FullyConnectedLayer(new int[] { inputs }, new int[] { outputs });
    this.inputs = inputs;
    this.outputs = outputs;
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return FullyConnectedReferenceLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { inputs } };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return fullyConnectedLayer;
  }

  public static @com.simiacryptus.ref.lang.RefAware class Basic extends FullyConnectedLayerTest {
    public Basic() {
      super(3, 3);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Basic addRef() {
      return (Basic) super.addRef();
    }

    public static @SuppressWarnings("unused") Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") FullyConnectedLayerTest addRef() {
    return (FullyConnectedLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") FullyConnectedLayerTest[] addRefs(FullyConnectedLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedLayerTest::addRef)
        .toArray((x) -> new FullyConnectedLayerTest[x]);
  }

  public static @SuppressWarnings("unused") FullyConnectedLayerTest[][] addRefs(FullyConnectedLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedLayerTest::addRefs)
        .toArray((x) -> new FullyConnectedLayerTest[x][]);
  }

  //  /**
  //   * The type BigTests.
  //   */
  //  public static class BigTests extends FullyConnectedLayerTest {
  //    /**
  //     * Instantiates a new BigTests.
  //     */
  //    public BigTests() {
  //      super(25088, 4096);
  //      validateDifferentials = false;
  //    }
  //
  //    @Override
  //    public Class<? extends LayerBase> getReferenceLayerClass() {
  //      return null;
  //    }
  //
  //  }

}
