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
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import java.util.Random;

public @com.simiacryptus.ref.lang.RefAware class SumInputsLayerTest {
  public static @com.simiacryptus.ref.lang.RefAware class N1Test extends LayerTestBase {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3 }, { 1 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new SumInputsLayer();
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][] { { 100 }, { 1 } };
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

  public static @com.simiacryptus.ref.lang.RefAware class NNTest extends LayerTestBase {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 3 }, { 3 } };
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new SumInputsLayer();
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][] { { 100 }, { 100 } };
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

  public static @com.simiacryptus.ref.lang.RefAware class OnePlusOne extends LayerTestBase {

    public OnePlusOne() {
      super();
    }

    @Override
    public Layer getReferenceLayer() {
      return null;
    }

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return SumInputsLayer.class;
    }

    @Nonnull
    @Override
    public Layer getLayer(int[][] inputSize, Random random) {
      @Nonnull
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.add(new SumInputsLayer(), input, input);
      return network;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][] { { 1, 1, 1 } };
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(random);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") OnePlusOne addRef() {
      return (OnePlusOne) super.addRef();
    }

    public static @SuppressWarnings("unused") OnePlusOne[] addRefs(OnePlusOne[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(OnePlusOne::addRef)
          .toArray((x) -> new OnePlusOne[x]);
    }

  }
}
