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

public abstract @com.simiacryptus.ref.lang.RefAware
class LinearActivationLayerTest extends ActivationLayerTestBase {
  public LinearActivationLayerTest() {
    super(new LinearActivationLayer());
  }

  public static @SuppressWarnings("unused")
  LinearActivationLayerTest[] addRefs(LinearActivationLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(LinearActivationLayerTest::addRef)
        .toArray((x) -> new LinearActivationLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  LinearActivationLayerTest[][] addRefs(LinearActivationLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(LinearActivationLayerTest::addRefs)
        .toArray((x) -> new LinearActivationLayerTest[x][]);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  LinearActivationLayerTest addRef() {
    return (LinearActivationLayerTest) super.addRef();
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Basic extends LinearActivationLayerTest {
    public static @SuppressWarnings("unused")
    Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Basic addRef() {
      return (Basic) super.addRef();
    }

    @Override
    protected Layer lossLayer() {
      return new EntropyLossLayer();
    }
  }

}
