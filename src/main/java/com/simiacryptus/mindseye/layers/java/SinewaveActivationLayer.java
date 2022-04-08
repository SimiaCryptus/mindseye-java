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
 * This class represents a sinewave activation layer.
 *
 * @author John Doe
 * @version 1.0
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public final class SinewaveActivationLayer extends SimpleActivationLayer<SinewaveActivationLayer> {

  private boolean balanced = true;

  /**
   * Instantiates a new Sinewave activation layer.
   */
  public SinewaveActivationLayer() {
  }

  /**
   * Instantiates a new Sinewave activation layer.
   *
   * @param id the id
   */
  protected SinewaveActivationLayer(@Nonnull final JsonObject id) {
    super(id);
    balanced = id.get("balanced").getAsBoolean();
  }

  /**
   * Returns true if the tree is balanced.
   *
   * @return true if the tree is balanced
   * @docgenVersion 9
   */
  public boolean isBalanced() {
    return balanced;
  }

  /**
   * Sets the balanced field.
   *
   * @param balanced the new value for balanced
   * @docgenVersion 9
   */
  public void setBalanced(boolean balanced) {
    this.balanced = balanced;
  }

  /**
   * Creates a {@link SinewaveActivationLayer} from a JSON object.
   *
   * @param json the JSON object to use for creating the layer
   * @param rs   a map of character sequences to byte arrays
   * @return a new {@link SinewaveActivationLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static SinewaveActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SinewaveActivationLayer(json);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("balanced", balanced);
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
  SinewaveActivationLayer addRef() {
    return (SinewaveActivationLayer) super.addRef();
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
}
