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
import java.util.Map;
import com.simiacryptus.ref.wrappers.RefMap;

@SuppressWarnings("serial")
public final @com.simiacryptus.ref.lang.RefAware class SinewaveActivationLayer
    extends SimpleActivationLayer<SinewaveActivationLayer> {

  private boolean balanced = true;

  public SinewaveActivationLayer() {
  }

  protected SinewaveActivationLayer(@Nonnull final JsonObject id) {
    super(id);
    balanced = id.get("balanced").getAsBoolean();
  }

  public boolean isBalanced() {
    return balanced;
  }

  @Nonnull
  public SinewaveActivationLayer setBalanced(final boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @SuppressWarnings("unused")
  public static SinewaveActivationLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new SinewaveActivationLayer(json);
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("balanced", balanced);
    return json;
  }

  @Override
  protected final void eval(final double x, final double[] results) {
    double d = Math.cos(x);
    double f = Math.sin(x);
    if (!isBalanced()) {
      d = d / 2;
      f = (f + 1) / 2;
    }
    results[0] = f;
    results[1] = d;
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") SinewaveActivationLayer addRef() {
    return (SinewaveActivationLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") SinewaveActivationLayer[] addRefs(SinewaveActivationLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SinewaveActivationLayer::addRef)
        .toArray((x) -> new SinewaveActivationLayer[x]);
  }

  public static @SuppressWarnings("unused") SinewaveActivationLayer[][] addRefs(SinewaveActivationLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SinewaveActivationLayer::addRefs)
        .toArray((x) -> new SinewaveActivationLayer[x][]);
  }
}
