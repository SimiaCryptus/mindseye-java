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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public @RefAware
class RescaledSubnetLayer extends LayerBase {

  private final int scale;
  @Nullable
  private final Layer subnetwork;

  public RescaledSubnetLayer(final int scale, @org.jetbrains.annotations.Nullable final Layer subnetwork) {
    super();
    this.scale = scale;
    {
      Layer temp_11_0001 = subnetwork == null ? null : subnetwork.addRef();
      this.subnetwork = temp_11_0001 == null ? null : temp_11_0001.addRef();
      if (null != temp_11_0001)
        temp_11_0001.freeRef();
    }
    if (null != subnetwork)
      subnetwork.freeRef();
  }

  protected RescaledSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    scale = json.getAsJsonPrimitive("scale").getAsInt();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
    {
      Layer temp_11_0005 = Layer.fromJson(subnetwork, rs);
      Layer temp_11_0002 = subnetwork == null ? null : temp_11_0005.addRef();
      if (null != temp_11_0005)
        temp_11_0005.freeRef();
      this.subnetwork = temp_11_0002 == null ? null : temp_11_0002.addRef();
      if (null != temp_11_0002)
        temp_11_0002.freeRef();
    }
  }

  @SuppressWarnings("unused")
  public static RescaledSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new RescaledSubnetLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  RescaledSubnetLayer[] addRefs(RescaledSubnetLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayer::addRef)
        .toArray((x) -> new RescaledSubnetLayer[x]);
  }

  public static @SuppressWarnings("unused")
  RescaledSubnetLayer[][] addRefs(RescaledSubnetLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RescaledSubnetLayer::addRefs)
        .toArray((x) -> new RescaledSubnetLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    TensorList temp_11_0006 = inObj[0].getData();
    @Nonnull final int[] inputDims = temp_11_0006.getDimensions();
    if (null != temp_11_0006)
      temp_11_0006.freeRef();
    assert 3 == inputDims.length;
    if (1 == scale) {
      Result temp_11_0004 = subnetwork
          .eval(Result.addRefs(inObj));
      ReferenceCounting.freeRefs(inObj);
      return temp_11_0004;
    }

    @Nonnull final PipelineNetwork network = new PipelineNetwork();
    @Nullable final DAGNode condensed = network.add(new ImgReshapeLayer(scale, scale, false));
    RefUtil.freeRef(network.add(new ImgConcatLayer(),
        RefIntStream.range(0, scale * scale).mapToObj(RefUtil.wrapInterface(
            (IntFunction<? extends InnerNode>) subband -> {
              @Nonnull final int[] select = new int[inputDims[2]];
              for (int i = 0; i < inputDims[2]; i++) {
                select[i] = subband * inputDims[2] + i;
              }
              return network.add(subnetwork == null ? null : subnetwork.addRef(),
                  network.add(new ImgBandSelectLayer(select), condensed == null ? null : condensed.addRef()));
            }, condensed == null ? null : condensed.addRef(), network == null ? null : network.addRef()))
            .toArray(i -> new DAGNode[i])));
    if (null != condensed)
      condensed.freeRef();
    RefUtil.freeRef(network.add(new ImgReshapeLayer(scale, scale, true)));
    Result temp_11_0003 = network
        .eval(Result.addRefs(inObj));
    ReferenceCounting.freeRefs(inObj);
    network.freeRef();
    return temp_11_0003;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("scale", scale);
    json.add("subnetwork", subnetwork.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  public void _free() {
    if (null != subnetwork)
      subnetwork.freeRef();
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  RescaledSubnetLayer addRef() {
    return (RescaledSubnetLayer) super.addRef();
  }

}
