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
import com.simiacryptus.mindseye.layers.ActivationLayerTestBase;
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;

public abstract class ReLuActivationLayerTest extends ActivationLayerTestBase {
  public ReLuActivationLayerTest() {
    super(new ReLuActivationLayer());
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ReLuActivationLayerTest addRef() {
    return (ReLuActivationLayerTest) super.addRef();
  }

  @Nonnull
  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }

  public static class Basic extends ReLuActivationLayerTest {

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
