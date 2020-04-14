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
import com.simiacryptus.ref.lang.MustCall;
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.jupiter.api.AfterEach;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Fully connected layer test.
 */
public abstract class FullyConnectedLayerTest extends LayerTestBase {

  @Nonnull
  @RefIgnore
  private final FullyConnectedLayer fullyConnectedLayer;
  private final int inputs;
  private final int outputs;

  /**
   * Instantiates a new Fully connected layer test.
   *
   * @param inputs  the inputs
   * @param outputs the outputs
   */
  protected FullyConnectedLayerTest(int inputs, int outputs) {
    FullyConnectedLayer temp_19_0001 = new FullyConnectedLayer(new int[]{inputs}, new int[]{outputs});
    fullyConnectedLayer = temp_19_0001.addRef();
    temp_19_0001.freeRef();
    this.inputs = inputs;
    this.outputs = outputs;
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return fullyConnectedLayer.addRef();
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return FullyConnectedReferenceLayer.class;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{inputs}};
  }

  /**
   * Cleanup.
   */
  @AfterEach
  @MustCall
  void cleanup() {
    if (null != fullyConnectedLayer)
      fullyConnectedLayer.freeRef();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends FullyConnectedLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(3, 3);
    }

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
