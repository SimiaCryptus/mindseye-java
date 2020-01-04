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
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class RescaledSubnetLayer extends LayerBase {

  private final int scale;
  @Nullable
  private final Layer subnetwork;

  public RescaledSubnetLayer(final int scale, @org.jetbrains.annotations.Nullable final Layer subnetwork) {
    super();
    this.scale = scale;
    this.subnetwork = subnetwork;
  }

  protected RescaledSubnetLayer(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json);
    scale = json.getAsJsonPrimitive("scale").getAsInt();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
    this.subnetwork = subnetwork == null ? null : Layer.fromJson(subnetwork, rs);
  }

  @SuppressWarnings("unused")
  public static RescaledSubnetLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new RescaledSubnetLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    @Nonnull
    final int[] inputDims = inObj[0].getData().getDimensions();
    assert 3 == inputDims.length;
    if (1 == scale)
      return subnetwork.eval(inObj);

    @Nonnull
    final PipelineNetwork network = new PipelineNetwork();
    @Nullable
    final DAGNode condensed = network.add(new ImgReshapeLayer(scale, scale, false));
    network.add(new ImgConcatLayer(),
        com.simiacryptus.ref.wrappers.RefIntStream.range(0, scale * scale).mapToObj(subband -> {
          @Nonnull
          final int[] select = new int[inputDims[2]];
          for (int i = 0; i < inputDims[2]; i++) {
            select[i] = subband * inputDims[2] + i;
          }
          return network.add(subnetwork, network.add(new ImgBandSelectLayer(select), condensed));
        }).toArray(i -> new DAGNode[i]));
    network.add(new ImgReshapeLayer(scale, scale, true));
    return network.eval(inObj);
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("scale", scale);
    json.add("subnetwork", subnetwork.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return new com.simiacryptus.ref.wrappers.RefArrayList<>();
  }

  public void _free() {
    super._free();
  }

  public @Override @SuppressWarnings("unused") RescaledSubnetLayer addRef() {
    return (RescaledSubnetLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") RescaledSubnetLayer[] addRefs(RescaledSubnetLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayer::addRef)
        .toArray((x) -> new RescaledSubnetLayer[x]);
  }

  public static @SuppressWarnings("unused") RescaledSubnetLayer[][] addRefs(RescaledSubnetLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayer::addRefs)
        .toArray((x) -> new RescaledSubnetLayer[x][]);
  }

}
