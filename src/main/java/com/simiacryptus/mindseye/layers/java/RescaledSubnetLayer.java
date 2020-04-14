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
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * The type Rescaled subnet layer.
 */
@SuppressWarnings("serial")
public class RescaledSubnetLayer extends LayerBase {

  private final int scale;
  @Nullable
  private final Layer subnetwork;

  /**
   * Instantiates a new Rescaled subnet layer.
   *
   * @param scale      the scale
   * @param subnetwork the subnetwork
   */
  public RescaledSubnetLayer(final int scale, @Nullable final Layer subnetwork) {
    super();
    this.scale = scale;
    this.subnetwork = subnetwork;
  }

  /**
   * Instantiates a new Rescaled subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected RescaledSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    scale = json.getAsJsonPrimitive("scale").getAsInt();
    this.subnetwork = Layer.fromJson(json.getAsJsonObject("inner"), rs);
  }

  /**
   * From json rescaled subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the rescaled subnet layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static RescaledSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new RescaledSubnetLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    TensorList in0data = inObj[0].getData();
    @Nonnull final int[] inputDims = in0data.getDimensions();
    in0data.freeRef();
    assert 3 == inputDims.length;
    if (1 == scale) {
      assert subnetwork != null;
      return subnetwork.eval(inObj);
    } else {
      @Nonnull final PipelineNetwork network = getNetwork(inputDims);
      Result result = network.eval(inObj);
      network.freeRef();
      return result;
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("scale", scale);
    assert subnetwork != null;
    json.add("inner", subnetwork.getJson(resources, dataSerializer));
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

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  RescaledSubnetLayer addRef() {
    return (RescaledSubnetLayer) super.addRef();
  }

  @NotNull
  private PipelineNetwork getNetwork(int[] inputDims) {
    int channels = inputDims[2];
    @Nonnull final PipelineNetwork network = new PipelineNetwork();
    @Nullable final DAGNode condensed = network.add(new ImgReshapeLayer(scale, scale, false));
    DAGNode[] nodes = RefIntStream.range(0, scale * scale)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends InnerNode>) subband -> {
          @Nonnull final int[] select = new int[channels];
          for (int i = 0; i < channels; i++) {
            select[i] = subband * channels + i;
          }
          return network.add(
              new WrapperLayer(subnetwork == null ? null : subnetwork.addRef()),
              network.add(new ImgBandSelectLayer(select), condensed.addRef()));
        }, condensed)).toArray(DAGNode[]::new);
    RefUtil.freeRef(network.add(new ImgConcatLayer(), nodes));
    RefUtil.freeRef(network.add(new ImgReshapeLayer(scale, scale, true)));
    return network;
  }

}
