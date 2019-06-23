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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class TargetValueLayer extends DAGNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TargetValueLayer.class);
  private final DAGNode head;
  private final DAGNode target;

  public TargetValueLayer(final double... values) {
    super(1);
    target = wrap(ValueLayer.wrap(new Tensor(values)));
    head = wrap(new MeanSqLossLayer(), getInput(0), target.addRef());
  }

  protected TargetValueLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    head = getNodeById(UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
    target = getNodeById(UUID.fromString(json.getAsJsonPrimitive("target").getAsString()));
  }

  public static Layer fromJson(@Nonnull final JsonObject inner, Map<CharSequence, byte[]> rs) {
    return new TargetValueLayer(inner, rs);
  }

  @Override
  protected void _free() {
    head.freeRef();
    target.freeRef();
    super._free();
  }

  @Override
  public DAGNode getHead() {
    head.addRef();
    return head;
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("target", target.getId().toString());
    return json;
  }

  @Nonnull
  public TargetValueLayer setTarget(final double... value) {
    target.<ValueLayer>getLayer().setData(new Tensor(value));
    return this;
  }
}
