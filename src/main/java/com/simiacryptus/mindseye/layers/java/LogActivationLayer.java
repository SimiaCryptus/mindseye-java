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
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;

@SuppressWarnings("serial")
public final @RefAware
class LogActivationLayer extends SimpleActivationLayer<LogActivationLayer> {

  public LogActivationLayer() {
  }

  protected LogActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static LogActivationLayer fromJson(final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new LogActivationLayer(json);
  }

  public static @SuppressWarnings("unused")
  LogActivationLayer[] addRefs(LogActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LogActivationLayer::addRef)
        .toArray((x) -> new LogActivationLayer[x]);
  }

  public static @SuppressWarnings("unused")
  LogActivationLayer[][] addRefs(LogActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LogActivationLayer::addRefs)
        .toArray((x) -> new LogActivationLayer[x][]);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  LogActivationLayer addRef() {
    return (LogActivationLayer) super.addRef();
  }

  @Override
  protected final void eval(final double x, final double[] results) {
    if (x < 0) {
      eval(-x, results);
      results[0] *= 1;
      results[1] *= -1;
    } else if (x > 0) {
      final double minDeriv = 0;
      final double d = 0 == x ? Double.NaN : 1 / x;
      final double f = 0 == x ? Double.NEGATIVE_INFINITY : Math.log(Math.abs(x));
      assert Double.isFinite(d);
      assert minDeriv <= Math.abs(d);
      results[0] = f;
      results[1] = d;
    } else {
      results[0] = 0;
      results[1] = 0;
    }
  }

}
