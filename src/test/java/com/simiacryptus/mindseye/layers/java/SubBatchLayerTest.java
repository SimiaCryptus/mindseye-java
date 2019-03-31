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
import javax.annotation.Nullable;
import java.util.Random;


public class SubBatchLayerTest extends LayerTestBase {

  private final Layer layer = SubBatchLayer.wrap(new SoftmaxLayer());

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {5}
    };
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer.copy();
  }


}