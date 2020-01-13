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
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public abstract class ImgViewLayerTest extends LayerTestBase {

  public ImgViewLayerTest() {
    validateBatchExecution = false;
  }

  public static @SuppressWarnings("unused") ImgViewLayerTest[] addRefs(ImgViewLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgViewLayerTest::addRef)
        .toArray((x) -> new ImgViewLayerTest[x]);
  }

  public static @SuppressWarnings("unused") ImgViewLayerTest[][] addRefs(ImgViewLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgViewLayerTest::addRefs)
        .toArray((x) -> new ImgViewLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 8, 8, 2 } };
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgViewLayerTest addRef() {
    return (ImgViewLayerTest) super.addRef();
  }

  public static class Basic extends ImgViewLayerTest {

    public static @SuppressWarnings("unused") Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgViewLayer(3, 2, 2, 3);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Basic addRef() {
      return (Basic) super.addRef();
    }
  }

  public static class Rotated extends ImgViewLayerTest {

    public static @SuppressWarnings("unused") Rotated[] addRefs(Rotated[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Rotated::addRef).toArray((x) -> new Rotated[x]);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgViewLayer temp_72_0002 = new ImgViewLayer(3, 2, 2, 3);
      ImgViewLayer temp_72_0001 = temp_72_0002.setRotationRadians(Math.PI / 2);
      if (null != temp_72_0002)
        temp_72_0002.freeRef();
      return temp_72_0001;
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Rotated addRef() {
      return (Rotated) super.addRef();
    }
  }

  public static class RotatedChannels extends ImgViewLayerTest {

    public static @SuppressWarnings("unused") RotatedChannels[] addRefs(RotatedChannels[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(RotatedChannels::addRef)
          .toArray((x) -> new RotatedChannels[x]);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      ImgViewLayer temp_72_0004 = new ImgViewLayer(3, 2, 2, 3);
      ImgViewLayer temp_72_0005 = temp_72_0004.setRotationRadians(Math.PI / 2);
      ImgViewLayer temp_72_0003 = temp_72_0005.setChannelSelector(2, -1);
      if (null != temp_72_0005)
        temp_72_0005.freeRef();
      if (null != temp_72_0004)
        temp_72_0004.freeRef();
      return temp_72_0003;
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") RotatedChannels addRef() {
      return (RotatedChannels) super.addRef();
    }
  }

}
