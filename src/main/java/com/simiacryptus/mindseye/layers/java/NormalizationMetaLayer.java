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
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class NormalizationMetaLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(NormalizationMetaLayer.class);

  public NormalizationMetaLayer() {
    super(1);
    add(new SqActivationLayer());
    add(new AvgReducerLayer());
    add(new AvgMetaLayer());
    add(new NthPowerActivationLayer().setPower(-0.5));
    add(new ProductInputsLayer(), getHead(), getInput(0));
  }

  protected NormalizationMetaLayer(@Nonnull final JsonObject json,
                                   com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  @SuppressWarnings("unused")
  public static NormalizationMetaLayer fromJson(@NotNull final JsonObject json,
                                                com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new NormalizationMetaLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  NormalizationMetaLayer[] addRefs(NormalizationMetaLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(NormalizationMetaLayer::addRef)
        .toArray((x) -> new NormalizationMetaLayer[x]);
  }

  public static @SuppressWarnings("unused")
  NormalizationMetaLayer[][] addRefs(NormalizationMetaLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(NormalizationMetaLayer::addRefs)
        .toArray((x) -> new NormalizationMetaLayer[x][]);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  NormalizationMetaLayer addRef() {
    return (NormalizationMetaLayer) super.addRef();
  }

}
