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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.ValueLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.ref.lang.RefAware;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class TargetValueLayer extends DAGNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TargetValueLayer.class);
  private final DAGNode head;
  private final DAGNode target;

  public TargetValueLayer(final double... values) {
    super(1);
    {
      DAGNode temp_05_0001 = add(new ValueLayer(new Tensor(values)));
      target = temp_05_0001 == null ? null : temp_05_0001.addRef();
      if (null != temp_05_0001)
        temp_05_0001.freeRef();
    }
    {
      DAGNode temp_05_0002 = add(new MeanSqLossLayer(), getInput(0),
          target == null ? null : target.addRef());
      head = temp_05_0002 == null ? null : temp_05_0002.addRef();
      if (null != temp_05_0002)
        temp_05_0002.freeRef();
    }
  }

  protected TargetValueLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    {
      DAGNode temp_05_0003 = getNodeById(
          UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
      head = temp_05_0003 == null ? null : temp_05_0003.addRef();
      if (null != temp_05_0003)
        temp_05_0003.freeRef();
    }
    {
      DAGNode temp_05_0004 = getNodeById(
          UUID.fromString(json.getAsJsonPrimitive("target").getAsString()));
      target = temp_05_0004 == null ? null : temp_05_0004.addRef();
      if (null != temp_05_0004)
        temp_05_0004.freeRef();
    }
  }

  @Override
  public DAGNode getHead() {
    return head == null ? null : head.addRef();
  }

  @Nonnull
  public TargetValueLayer setTarget(final double... value) {
    ValueLayer temp_05_0005 = target.<ValueLayer>getLayer();
    temp_05_0005.setData(new Tensor(value));
    if (null != temp_05_0005)
      temp_05_0005.freeRef();
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static Layer fromJson(@Nonnull final JsonObject inner, Map<CharSequence, byte[]> rs) {
    return new TargetValueLayer(inner, rs);
  }

  public static @SuppressWarnings("unused")
  TargetValueLayer[] addRefs(TargetValueLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TargetValueLayer::addRef)
        .toArray((x) -> new TargetValueLayer[x]);
  }

  public static @SuppressWarnings("unused")
  TargetValueLayer[][] addRefs(TargetValueLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TargetValueLayer::addRefs)
        .toArray((x) -> new TargetValueLayer[x][]);
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("target", target.getId().toString());
    return json;
  }

  public void _free() {
    if (null != target)
      target.freeRef();
    if (null != head)
      head.freeRef();
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  TargetValueLayer addRef() {
    return (TargetValueLayer) super.addRef();
  }
}
