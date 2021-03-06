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
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;

/**
 * The type Stochastic sampling subnet layer test.
 */
public abstract class StochasticSamplingSubnetLayerTest extends LayerTestBase {

  @Nonnull
  @Override
  public Layer getLayer() {
    PipelineNetwork subnetwork = new PipelineNetwork(1);
    RefUtil.freeRef(subnetwork.add(new ProductLayer(), subnetwork.getInput(0),
        subnetwork.add(new BinaryNoiseLayer(0.5), subnetwork.getInput(0))));

    StochasticSamplingSubnetLayer temp_38_0001 = new StochasticSamplingSubnetLayer(
        subnetwork.addRef(), 2);
    subnetwork.freeRef();
    return temp_38_0001;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{6, 6, 1}};
  }

  /**
   * The type Basic.
   */
  public static class Basic extends StochasticSamplingSubnetLayerTest {

  }

}
