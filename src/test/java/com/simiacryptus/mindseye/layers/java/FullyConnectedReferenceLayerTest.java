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
import java.util.Random;

public abstract class FullyConnectedReferenceLayerTest extends LayerTestBase {
  @Nonnull
  private final int[] outputDims;
  private final int[] inputDims;
  @Nonnull
  private final FullyConnectedReferenceLayer layer;


  public FullyConnectedReferenceLayerTest(int[] inputDims, @Nonnull int[] outputDims) {
    this.outputDims = outputDims;
    this.inputDims = inputDims;
    this.layer = new FullyConnectedReferenceLayer(getSmallDims(new Random())[0], outputDims).set(i -> random());
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        inputDims
    };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    layer.addRef();
    return layer;
  }

  public static class Basic extends FullyConnectedReferenceLayerTest {
    public Basic() {
      super(new int[]{2}, new int[]{2});
    }
  }

  public static class Image extends FullyConnectedReferenceLayerTest {
    public Image() {
      super(new int[]{3, 3, 3}, new int[]{2, 2, 4});
    }
  }

}
