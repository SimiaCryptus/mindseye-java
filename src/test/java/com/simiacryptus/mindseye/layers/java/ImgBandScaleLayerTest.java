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

public abstract class ImgBandScaleLayerTest extends LayerTestBase {

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 2, 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    ImgBandScaleLayer temp_76_0002 = new ImgBandScaleLayer(0.0, 0.0, 0.0);
    temp_76_0002.addWeights(this::random);
    ImgBandScaleLayer temp_76_0001 = temp_76_0002.addRef();
    temp_76_0002.freeRef();
    return temp_76_0001;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgBandScaleLayerTest addRef() {
    return (ImgBandScaleLayerTest) super.addRef();
  }

  public static class Basic extends ImgBandScaleLayerTest {

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }
  }

}
