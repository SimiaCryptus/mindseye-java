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
public class BoundedActivationLayer extends SimpleActivationLayer<BoundedActivationLayer> {

  private double maxValue = Double.POSITIVE_INFINITY;
  private double minValue = Double.NEGATIVE_INFINITY;

  public BoundedActivationLayer() {
    super();
  }

  protected BoundedActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  public double getMaxValue() {
    return maxValue;
  }

  @Nonnull
  public BoundedActivationLayer setMaxValue(final double maxValue) {
    this.maxValue = maxValue;
    return this.addRef();
  }

  public double getMinValue() {
    return minValue;
  }

  public BoundedActivationLayer setMinValue(double minValue) {
    this.minValue = minValue;
    return this.addRef();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static BoundedActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    @Nonnull
    final BoundedActivationLayer obj = new BoundedActivationLayer(json);
    obj.maxValue = json.get("maxValue").getAsDouble();
    return obj;
  }

  public static @SuppressWarnings("unused") BoundedActivationLayer[] addRefs(BoundedActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BoundedActivationLayer::addRef)
        .toArray((x) -> new BoundedActivationLayer[x]);
  }

  public static @SuppressWarnings("unused") BoundedActivationLayer[][] addRefs(BoundedActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BoundedActivationLayer::addRefs)
        .toArray((x) -> new BoundedActivationLayer[x][]);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("maxValue", maxValue);
    return json;
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") BoundedActivationLayer addRef() {
    return (BoundedActivationLayer) super.addRef();
  }

  @Override
  protected void eval(final double x, final double[] results) {
    final double d = x > maxValue || x < getMinValue() ? 0 : 1;
    final double f = x > maxValue ? maxValue : x < getMinValue() ? getMinValue() : x;
    assert Double.isFinite(d);
    results[0] = f;
    results[1] = d;
  }
}
