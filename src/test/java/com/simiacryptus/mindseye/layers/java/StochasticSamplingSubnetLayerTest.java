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
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware
class StochasticSamplingSubnetLayerTest extends LayerTestBase {

  public static @SuppressWarnings("unused")
  StochasticSamplingSubnetLayerTest[] addRefs(
      StochasticSamplingSubnetLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StochasticSamplingSubnetLayerTest::addRef)
        .toArray((x) -> new StochasticSamplingSubnetLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  StochasticSamplingSubnetLayerTest[][] addRefs(
      StochasticSamplingSubnetLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StochasticSamplingSubnetLayerTest::addRefs)
        .toArray((x) -> new StochasticSamplingSubnetLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{6, 6, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    PipelineNetwork subnetwork = new PipelineNetwork(1);
    subnetwork.add(new ProductLayer(), subnetwork.getInput(0),
        subnetwork.add(new BinaryNoiseLayer(0.5), subnetwork.getInput(0)));

    return new StochasticSamplingSubnetLayer(subnetwork, 2);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  StochasticSamplingSubnetLayerTest addRef() {
    return (StochasticSamplingSubnetLayerTest) super.addRef();
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Basic extends StochasticSamplingSubnetLayerTest {

    public static @SuppressWarnings("unused")
    Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }
  }

}
