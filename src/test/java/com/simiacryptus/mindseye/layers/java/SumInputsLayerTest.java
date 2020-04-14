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
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;

/**
 * The type Sum inputs layer test.
 */
public class SumInputsLayerTest {
  /**
   * The type N 1 test.
   */
  public static class N1Test extends LayerTestBase {

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{100}, {1}};
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      return new SumInputsLayer();
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{3}, {1}};
    }

  }

  /**
   * The type Nn test.
   */
  public static class NNTest extends LayerTestBase {

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return new int[][]{{100}, {100}};
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      return new SumInputsLayer();
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{3}, {3}};
    }

  }

  /**
   * The type One plus one.
   */
  public static class OnePlusOne extends LayerTestBase {

    /**
     * Instantiates a new One plus one.
     */
    public OnePlusOne() {
      super();
    }

    @Nonnull
    @Override
    public int[][] getLargeDims() {
      return getSmallDims();
    }

    @Nonnull
    @Override
    public Layer getLayer() {
      @Nonnull
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      RefUtil.freeRef(network.add(new SumInputsLayer(), input.addRef(),
          input.addRef()));
      input.freeRef();
      return network;
    }

    @Override
    public Layer getReferenceLayer() {
      return null;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{1, 1, 1}};
    }

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return SumInputsLayer.class;
    }

  }
}
