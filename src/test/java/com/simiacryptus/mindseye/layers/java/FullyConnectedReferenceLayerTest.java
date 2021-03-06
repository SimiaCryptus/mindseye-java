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

/**
 * The type Fully connected reference layer test.
 */
public abstract class FullyConnectedReferenceLayerTest extends LayerTestBase {
  @Nonnull
  private final int[] outputDims;
  private final int[] inputDims;
  @Nonnull
  @RefIgnore
  private final FullyConnectedReferenceLayer layer;

  /**
   * Instantiates a new Fully connected reference layer test.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public FullyConnectedReferenceLayerTest(int[] inputDims, @Nonnull int[] outputDims) {
    this.outputDims = outputDims;
    this.inputDims = inputDims;
    FullyConnectedReferenceLayer temp_00_0002 = new FullyConnectedReferenceLayer(getSmallDims()[0],
        outputDims);
    temp_00_0002.set(i -> random());
    FullyConnectedReferenceLayer temp_00_0001 = temp_00_0002.addRef();
    temp_00_0002.freeRef();
    this.layer = temp_00_0001.addRef();
    temp_00_0001.freeRef();
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return layer.addRef();
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{inputDims};
  }

  /**
   * Cleanup.
   */
  @AfterEach
  @MustCall
  void cleanup() {
    if (null != layer)
      layer.freeRef();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends FullyConnectedReferenceLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(new int[]{2}, new int[]{2});
    }

  }

  /**
   * The type Image.
   */
  public static class Image extends FullyConnectedReferenceLayerTest {
    /**
     * Instantiates a new Image.
     */
    public Image() {
      super(new int[]{3, 3, 3}, new int[]{2, 2, 4});
    }

  }

}
