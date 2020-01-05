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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;

@SuppressWarnings("serial")
public @RefAware
class NormalizationMetaLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(NormalizationMetaLayer.class);

  public NormalizationMetaLayer() {
    super(1);
    RefUtil.freeRef(add(new SqActivationLayer()));
    RefUtil.freeRef(add(new AvgReducerLayer()));
    RefUtil.freeRef(add(new AvgMetaLayer()));
    NthPowerActivationLayer temp_74_0001 = new NthPowerActivationLayer();
    RefUtil.freeRef(add(temp_74_0001.setPower(-0.5)));
    if (null != temp_74_0001)
      temp_74_0001.freeRef();
    RefUtil.freeRef(add(new ProductInputsLayer(), getHead(), getInput(0)));
  }

  protected NormalizationMetaLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  @SuppressWarnings("unused")
  public static NormalizationMetaLayer fromJson(@NotNull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new NormalizationMetaLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  NormalizationMetaLayer[] addRefs(NormalizationMetaLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(NormalizationMetaLayer::addRef)
        .toArray((x) -> new NormalizationMetaLayer[x]);
  }

  public static @SuppressWarnings("unused")
  NormalizationMetaLayer[][] addRefs(NormalizationMetaLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(NormalizationMetaLayer::addRefs)
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
