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
public final class SigmoidActivationLayer extends SimpleActivationLayer<SigmoidActivationLayer> {

  private static final double MIN_X = -20;
  private static final double MAX_X = -SigmoidActivationLayer.MIN_X;
  private static final double MAX_F = Math.exp(SigmoidActivationLayer.MAX_X);
  private static final double MIN_F = Math.exp(SigmoidActivationLayer.MIN_X);
  private boolean balanced = true;

  public SigmoidActivationLayer() {
  }

  protected SigmoidActivationLayer(@Nonnull final JsonObject id) {
    super(id);
    balanced = id.get("balanced").getAsBoolean();
  }

  public boolean isBalanced() {
    return balanced;
  }

  public void setBalanced(boolean balanced) {
    this.balanced = balanced;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SigmoidActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SigmoidActivationLayer(json);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("balanced", balanced);
    return json;
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SigmoidActivationLayer addRef() {
    return (SigmoidActivationLayer) super.addRef();
  }

  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double ex = exp(x);
    final double ex1 = 1 + ex;
    double d = ex / (ex1 * ex1);
    double f = 1 / (1 + 1. / ex);
    // double d = f * (1 - f);
    if (!Double.isFinite(d) || d < minDeriv) {
      d = minDeriv;
    }
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    if (isBalanced()) {
      d = 2 * d;
      f = 2 * f - 1;
    }
    results[0] = f;
    results[1] = d;
  }

  private double exp(final double x) {
    if (x < SigmoidActivationLayer.MIN_X) {
      return SigmoidActivationLayer.MIN_F;
    }
    if (x > SigmoidActivationLayer.MAX_X) {
      return SigmoidActivationLayer.MAX_F;
    }
    return Math.exp(x);
  }
}
