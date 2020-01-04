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

public @com.simiacryptus.ref.lang.RefAware
class SignReducerLayerTest {
  public static @com.simiacryptus.ref.lang.RefAware
  class Normal extends LayerTestBase {

    public static @SuppressWarnings("unused")
    Normal[] addRefs(Normal[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Normal::addRef).toArray((x) -> new Normal[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{3}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new SignReducerLayer();
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Normal addRef() {
      return (Normal) super.addRef();
    }
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Basic extends SignReducerLayerTest {
  }

}
