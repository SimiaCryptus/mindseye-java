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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;

@SuppressWarnings("serial")
public final class BinaryEntropyActivationLayer extends SimpleActivationLayer<BinaryEntropyActivationLayer> {

  public BinaryEntropyActivationLayer() {
  }

  protected BinaryEntropyActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static BinaryEntropyActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BinaryEntropyActivationLayer(json);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  BinaryEntropyActivationLayer[] addRefs(
      @Nullable BinaryEntropyActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BinaryEntropyActivationLayer::addRef)
        .toArray((x) -> new BinaryEntropyActivationLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  BinaryEntropyActivationLayer[][] addRefs(
      @Nullable BinaryEntropyActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BinaryEntropyActivationLayer::addRefs)
        .toArray((x) -> new BinaryEntropyActivationLayer[x][]);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BinaryEntropyActivationLayer addRef() {
    return (BinaryEntropyActivationLayer) super.addRef();
  }

  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double d = 0 >= x ? Double.NaN : Math.log(x) - Math.log(1 - x);
    final double f = 0 >= x || 1 <= x ? Double.POSITIVE_INFINITY : x * Math.log(x) + (1 - x) * Math.log(1 - x);
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }

}
