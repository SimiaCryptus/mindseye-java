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
import com.simiacryptus.mindseye.test.LayerTestBase;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Img band bias layer test.
 */
public abstract class CoordinateReassemblyLayerTest extends LayerTestBase {

  public CoordinateReassemblyLayerTest() {
    this.testingBatchSize = 1;
    this.sublayerTesting = false;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{
        {2, 2, 3}
    };
  }

  @Override
  protected @Nonnull int[][] getLargeDims() {
    return new int[][]{
        {10, 10, 3}
    };
  }

  @Override
  public void batchingTest() {
    // Disable Test
  }

  /**
   * The type Basic.
   */
  public static class Basic extends CoordinateReassemblyLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      PipelineNetwork network = new PipelineNetwork();
      network.add(new CoordinateDisassemblyLayer()).freeRef();
      network.add(new CoordinateAssemblyLayer(), network.getHead(), network.getInput(0)).freeRef();
      return network;
    }

    @Override
    protected @Nullable Layer getReferenceLayer() {
      return new PipelineNetwork(1);
    }

  }

  /**
   * The type Basic.
   */
  public static class IgnoreInputColors extends CoordinateReassemblyLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      PipelineNetwork network = new PipelineNetwork();
      network.add(new CoordinateDisassemblyLayer(false)).freeRef();
      network.add(new FullyConnectedLayer(new int[] {2}, new int[] {3})).freeRef();
      network.add(new CoordinateAssemblyLayer(false), network.getHead(), network.getInput(0)).freeRef();
      return network;
    }

  }

}
