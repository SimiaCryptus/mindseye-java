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
 * This class represents an activation layer in a neural network.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public final class AbsActivationLayer extends SimpleActivationLayer<AbsActivationLayer> {

  /**
   * Instantiates a new Abs activation layer.
   */
  public AbsActivationLayer() {
  }

  /**
   * Instantiates a new Abs activation layer.
   *
   * @param id the id
   */
  protected AbsActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * Returns an {@link AbsActivationLayer} from a JSON object.
   *
   * @param json the JSON object to parse
   * @param rs   a map of character sequences to byte arrays
   * @return the {@link AbsActivationLayer} object
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static AbsActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AbsActivationLayer(json);
  }

  /**
   * @param resources      the resources to be used
   * @param dataSerializer the dataSerializer to be used
   * @return a JSON object
   * @docgenVersion 9
   */
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
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

  /**
   * Adds a reference to this object.
   *
   * @return A reference to this object.
   * @docgenVersion 9
   */
  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  AbsActivationLayer addRef() {
    return (AbsActivationLayer) super.addRef();
  }

  /**
   * Evaluates the function and its derivative at the argument x.
   *
   * @param x       the argument
   * @param results an array where the function value and derivative value are stored
   * @throws IllegalArgumentException if x is not finite
   * @throws IllegalArgumentException if results is not a 2-element array
   * @docgenVersion 9
   */
  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double d = x < 0 ? -1 : 1;
    final double f = x < 0 ? -x : x;
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }

}
