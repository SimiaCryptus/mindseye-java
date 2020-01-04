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
import com.simiacryptus.util.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ReshapeLayer extends LayerBase {
  private static final Logger log = LoggerFactory.getLogger(ReshapeLayer.class);
  @Nullable
  public final int[] outputDims;

  private ReshapeLayer() {
    outputDims = null;
  }

  public ReshapeLayer(@Nonnull final int... outputDims) {
    this.outputDims = com.simiacryptus.ref.wrappers.RefArrays.copyOf(outputDims, outputDims.length);
  }

  protected ReshapeLayer(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
  }

  @SuppressWarnings("unused")
  public static ReshapeLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ReshapeLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    TensorList data = inObj[0].getData();
    @Nonnull
    int[] inputDims = data.getDimensions();
    ReshapedTensorList reshapedTensorList = new ReshapedTensorList(data, outputDims);
    return new Result(reshapedTensorList, (DeltaSet<UUID> buffer, TensorList delta) -> {
      @Nonnull
      ReshapedTensorList tensorList = new ReshapedTensorList(delta, inputDims);
      inObj[0].accumulate(buffer, tensorList);
    }) {

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }

      public void _free() {
      }
    };

  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ReshapeLayer addRef() {
    return (ReshapeLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ReshapeLayer[] addRefs(ReshapeLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ReshapeLayer::addRef)
        .toArray((x) -> new ReshapeLayer[x]);
  }

  public static @SuppressWarnings("unused") ReshapeLayer[][] addRefs(ReshapeLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ReshapeLayer::addRefs)
        .toArray((x) -> new ReshapeLayer[x][]);
  }

}
