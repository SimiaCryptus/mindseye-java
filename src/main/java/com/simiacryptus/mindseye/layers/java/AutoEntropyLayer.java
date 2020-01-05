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
import com.simiacryptus.mindseye.network.DAGNode;
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
class AutoEntropyLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AutoEntropyLayer.class);

  public AutoEntropyLayer() {
    super(1);
    DAGNode input = getInput(0);
    RefUtil.freeRef(
        add(new EntropyLossLayer(), input == null ? null : input.addRef(), input == null ? null : input.addRef()));
    if (null != input)
      input.freeRef();
  }

  protected AutoEntropyLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  @SuppressWarnings("unused")
  public static AutoEntropyLayer fromJson(@NotNull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AutoEntropyLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  AutoEntropyLayer[] addRefs(AutoEntropyLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AutoEntropyLayer::addRef)
        .toArray((x) -> new AutoEntropyLayer[x]);
  }

  public static @SuppressWarnings("unused")
  AutoEntropyLayer[][] addRefs(AutoEntropyLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AutoEntropyLayer::addRefs)
        .toArray((x) -> new AutoEntropyLayer[x][]);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  AutoEntropyLayer addRef() {
    return (AutoEntropyLayer) super.addRef();
  }

}
