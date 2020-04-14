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
 * The type Sub batch layer test.
 */
public class SubBatchLayerTest extends LayerTestBase {

  @Nullable
  @RefIgnore
  private final Layer layer = SubBatchLayer.wrap(new SoftmaxLayer());

  @Nonnull
  @Override
  public Layer getLayer() {
    assert layer != null;
    return layer.copy();
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{5}};
  }

  /**
   * Cleanup.
   */
  @AfterEach
  @MustCall
  public void cleanup() {
    if (null != layer)
      layer.freeRef();
  }

}
