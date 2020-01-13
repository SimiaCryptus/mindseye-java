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
import com.simiacryptus.ref.lang.RefAware;

import java.util.Arrays;

public abstract class AbsActivationLayerTest extends ActivationLayerTestBase {
  public AbsActivationLayerTest() {
    super(new AbsActivationLayer());
  }

  public static @SuppressWarnings("unused") AbsActivationLayerTest[] addRefs(AbsActivationLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AbsActivationLayerTest::addRef)
        .toArray((x) -> new AbsActivationLayerTest[x]);
  }

  public static @SuppressWarnings("unused") AbsActivationLayerTest[][] addRefs(AbsActivationLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AbsActivationLayerTest::addRefs)
        .toArray((x) -> new AbsActivationLayerTest[x][]);
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") AbsActivationLayerTest addRef() {
    return (AbsActivationLayerTest) super.addRef();
  }

  public static class Basic extends AbsActivationLayerTest {

    public static @SuppressWarnings("unused") Basic[] addRefs(Basic[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Basic::addRef).toArray((x) -> new Basic[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") Basic addRef() {
      return (Basic) super.addRef();
    }

    @Override
    protected Layer lossLayer() {
      return new EntropyLossLayer();
    }
  }
}
