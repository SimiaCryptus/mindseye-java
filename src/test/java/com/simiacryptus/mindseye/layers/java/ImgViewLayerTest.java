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

public abstract class ImgViewLayerTest extends LayerTestBase {

  public ImgViewLayerTest() {
    validateBatchExecution = false;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{8, 8, 2}};
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgViewLayerTest addRef() {
    return (ImgViewLayerTest) super.addRef();
  }

  public static class Basic extends ImgViewLayerTest {

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgViewLayer(3, 2, 2, 3);
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }
  }

  public static class Rotated extends ImgViewLayerTest {

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgViewLayer temp_72_0002 = new ImgViewLayer(3, 2, 2, 3);
      temp_72_0002.setRotationRadians(Math.PI / 2);
      ImgViewLayer temp_72_0001 = temp_72_0002.addRef();
      temp_72_0002.freeRef();
      return temp_72_0001;
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Rotated addRef() {
      return (Rotated) super.addRef();
    }
  }

  public static class RotatedChannels extends ImgViewLayerTest {

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgViewLayer temp_72_0004 = new ImgViewLayer(3, 2, 2, 3);
      temp_72_0004.setRotationRadians(Math.PI / 2);
      ImgViewLayer temp_72_0005 = temp_72_0004.addRef();
      temp_72_0005.setChannelSelector(new int[]{2, -1});
      ImgViewLayer temp_72_0003 = temp_72_0005.addRef();
      temp_72_0005.freeRef();
      temp_72_0004.freeRef();
      return temp_72_0003;
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    RotatedChannels addRef() {
      return (RotatedChannels) super.addRef();
    }
  }

}
