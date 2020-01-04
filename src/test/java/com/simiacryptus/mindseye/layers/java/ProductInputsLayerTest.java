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
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware class ProductInputsLayerTest extends LayerTestBase {
  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return new TrainingTester() {

      @Override
      protected Layer lossLayer() {
        return ProductInputsLayerTest.this.lossLayer();
      }

      public @SuppressWarnings("unused") void _free() {
      }
    }.setRandomizationMode(TrainingTester.RandomizationMode.Random);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ProductInputsLayer();
  }

  public static @com.simiacryptus.ref.lang.RefAware class N1Test extends ProductInputsLayerTest {
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3 }, { 1 } };
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") N1Test addRef() {
      return (N1Test) super.addRef();
    }

    public static @SuppressWarnings("unused") N1Test[] addRefs(N1Test[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(N1Test::addRef).toArray((x) -> new N1Test[x]);
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware class NNNTest extends ProductInputsLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3 }, { 3 }, { 3 } };
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") NNNTest addRef() {
      return (NNNTest) super.addRef();
    }

    public static @SuppressWarnings("unused") NNNTest[] addRefs(NNNTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(NNNTest::addRef)
          .toArray((x) -> new NNNTest[x]);
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware class NNTest extends ProductInputsLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3 }, { 3 } };
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") NNTest addRef() {
      return (NNTest) super.addRef();
    }

    public static @SuppressWarnings("unused") NNTest[] addRefs(NNTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(NNTest::addRef).toArray((x) -> new NNTest[x]);
    }
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ProductInputsLayerTest addRef() {
    return (ProductInputsLayerTest) super.addRef();
  }

  public static @SuppressWarnings("unused") ProductInputsLayerTest[] addRefs(ProductInputsLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ProductInputsLayerTest::addRef)
        .toArray((x) -> new ProductInputsLayerTest[x]);
  }

  public static @SuppressWarnings("unused") ProductInputsLayerTest[][] addRefs(ProductInputsLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ProductInputsLayerTest::addRefs)
        .toArray((x) -> new ProductInputsLayerTest[x][]);
  }
}
