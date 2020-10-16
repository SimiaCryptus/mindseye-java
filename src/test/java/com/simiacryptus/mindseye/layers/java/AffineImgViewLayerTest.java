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
import org.junit.jupiter.api.Disabled;

import javax.annotation.Nonnull;

/**
 * The type Img view layer test.
 */
public abstract class AffineImgViewLayerTest extends LayerTestBase {

  /**
   * Instantiates a new Img view layer test.
   */
  public AffineImgViewLayerTest() {
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{8, 8, 2}};
  }

  @Override
  @Disabled
  public void batchingTest() {
    super.batchingTest();
  }

  /**
   * The type Basic.
   */
  public static class Basic extends AffineImgViewLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      return new AffineImgViewLayer(3, 2, 2, 3);
    }

  }

  /**
   * The type Rotated.
   */
  public static class Rotated extends AffineImgViewLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      AffineImgViewLayer imgViewLayer = new AffineImgViewLayer(3, 2, 2, 3);
      imgViewLayer.setRotationRadians(Math.PI / 2);
      return imgViewLayer;
    }

  }

  /**
   * The type Rotated channels.
   */
  public static class RotatedChannels extends AffineImgViewLayerTest {

    @Nonnull
    @Override
    public Layer getLayer() {
      AffineImgViewLayer imgViewLayer = new AffineImgViewLayer(3, 2, 2, 3);
      imgViewLayer.setRotationRadians(Math.PI / 2);
      imgViewLayer.setChannelSelector(new int[]{2, -1});
      return imgViewLayer;
    }

  }

}
