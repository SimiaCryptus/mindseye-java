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
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public abstract class ProductInputsLayerTest extends LayerTestBase {
  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    TrainingTester temp_67_0002 = new TrainingTester() {

      public @SuppressWarnings("unused") void _free() {
      }

      @Override
      protected Layer lossLayer() {
        return ProductInputsLayerTest.this.lossLayer();
      }
    };
    TrainingTester temp_67_0001 = temp_67_0002.setRandomizationMode(TrainingTester.RandomizationMode.Random);
    if (null != temp_67_0002)
      temp_67_0002.freeRef();
    return temp_67_0001;
  }

  public static @SuppressWarnings("unused") ProductInputsLayerTest[] addRefs(ProductInputsLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductInputsLayerTest::addRef)
        .toArray((x) -> new ProductInputsLayerTest[x]);
  }

  public static @SuppressWarnings("unused") ProductInputsLayerTest[][] addRefs(ProductInputsLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductInputsLayerTest::addRefs)
        .toArray((x) -> new ProductInputsLayerTest[x][]);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ProductInputsLayer();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ProductInputsLayerTest addRef() {
    return (ProductInputsLayerTest) super.addRef();
  }

  public static class N1Test extends ProductInputsLayerTest {
    public static @SuppressWarnings("unused") N1Test[] addRefs(N1Test[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(N1Test::addRef).toArray((x) -> new N1Test[x]);
    }

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
  }

  public static class NNNTest extends ProductInputsLayerTest {

    public static @SuppressWarnings("unused") NNNTest[] addRefs(NNNTest[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(NNNTest::addRef).toArray((x) -> new NNNTest[x]);
    }

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
  }

  public static class NNTest extends ProductInputsLayerTest {

    public static @SuppressWarnings("unused") NNTest[] addRefs(NNTest[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(NNTest::addRef).toArray((x) -> new NNTest[x]);
    }

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
  }
}
