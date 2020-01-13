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
public class EntropyLayer extends SimpleActivationLayer<EntropyLayer> {

  public EntropyLayer() {
    super();
  }

  protected EntropyLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static EntropyLayer fromJson(final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new EntropyLayer(json);
  }

  public static @SuppressWarnings("unused") EntropyLayer[] addRefs(EntropyLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(EntropyLayer::addRef).toArray((x) -> new EntropyLayer[x]);
  }

  public static @SuppressWarnings("unused") EntropyLayer[][] addRefs(EntropyLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(EntropyLayer::addRefs)
        .toArray((x) -> new EntropyLayer[x][]);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") EntropyLayer addRef() {
    return (EntropyLayer) super.addRef();
  }

  @Override
  protected void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    double d;
    double f;
    if (0. == x) {
      d = 0;
      f = 0;
    } else {
      final double log = Math.log(Math.abs(x));
      d = -(1 + log);
      f = -x * log;
    }
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
}
