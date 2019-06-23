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

  @Nonnull
  public static BoundedActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    @Nonnull final BoundedActivationLayer obj = new BoundedActivationLayer(json);
    obj.maxValue = json.get("maxValue").getAsDouble();
    return obj;
  }

  @Override
  protected void eval(final double x, final double[] results) {
    final double d = x > maxValue || x < getMinValue() ? 0 : 1;
    final double f = x > maxValue ? maxValue : x < getMinValue() ? getMinValue() : x;
    assert Double.isFinite(d);
    results[0] = f;
    results[1] = d;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("maxValue", maxValue);
    return json;
  }

  public double getMaxValue() {
    return maxValue;
  }

  @Nonnull
  public BoundedActivationLayer setMaxValue(final double maxValue) {
    this.maxValue = maxValue;
    return this;
  }

  public double getMinValue() {
    return minValue;
  }

  public BoundedActivationLayer setMinValue(double minValue) {
    this.minValue = minValue;
    return this;
  }
}
