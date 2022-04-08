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

/**
 * This class represents a bounded activation layer.
 *
 * @author John Doe
 * @version 1.0
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class BoundedActivationLayer extends SimpleActivationLayer<BoundedActivationLayer> {

  private double maxValue = Double.POSITIVE_INFINITY;
  private double minValue = Double.NEGATIVE_INFINITY;

  /**
   * Instantiates a new Bounded activation layer.
   */
  public BoundedActivationLayer() {
    super();
  }

  /**
   * Instantiates a new Bounded activation layer.
   *
   * @param id the id
   */
  protected BoundedActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * Returns the max value.
   *
   * @return the max value
   * @docgenVersion 9
   */
  public double getMaxValue() {
    return maxValue;
  }

  /**
   * Sets the maximum value for the range.
   *
   * @param maxValue the maximum value for the range
   * @docgenVersion 9
   */
  public void setMaxValue(double maxValue) {
    this.maxValue = maxValue;
  }

  /**
   * Returns the minimum value of the data set.
   *
   * @return the minimum value of the data set
   * @docgenVersion 9
   */
  public double getMinValue() {
    return minValue;
  }

  /**
   * Sets the minimum value for the range.
   *
   * @param minValue the minimum value for the range
   * @docgenVersion 9
   */
  public void setMinValue(double minValue) {
    this.minValue = minValue;
  }

  /**
   * Creates a {@link BoundedActivationLayer} from a JSON object.
   *
   * @param json the JSON object to use
   * @param rs   the map of character sequences to byte arrays
   * @return the created {@link BoundedActivationLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static BoundedActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    @Nonnull final BoundedActivationLayer obj = new BoundedActivationLayer(json);
    obj.maxValue = json.get("maxValue").getAsDouble();
    return obj;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("maxValue", maxValue);
    return json;
  }

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BoundedActivationLayer addRef() {
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
