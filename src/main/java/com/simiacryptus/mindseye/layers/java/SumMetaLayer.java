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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class SumMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumMetaLayer.class);
  @Nullable
  private Tensor lastResult;
  private int minBatches = 1;

  public SumMetaLayer() {
  }

  protected SumMetaLayer(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources) {
    super(json);
    lastResult = Tensor.fromJson(json.get("lastResult"), resources);
    minBatches = json.get("minBatches").getAsInt();
  }

  public int getMinBatches() {
    return minBatches;
  }

  @Nonnull
  public SumMetaLayer setMinBatches(final int minBatches) {
    this.minBatches = minBatches;
    return this;
  }

  @SuppressWarnings("unused")
  public static SumMetaLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new SumMetaLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (1 != inObj.length)
      throw new IllegalArgumentException();
    final Result input = inObj[0];
    TensorList inputData = input.getData();
    final int itemCnt = inputData.length();
    if (null == lastResult || minBatches < itemCnt) {
      @Nonnull
      final ToDoubleFunction<Coordinate> f = (c) -> com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt)
          .mapToDouble(dataIndex -> {
            Tensor tensor = inputData.get(dataIndex);
            return tensor.get(c);
          }).sum();
      lastResult = inputData.get(0).mapCoords(f);
    }
    return new Result(new TensorArray(lastResult),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
          if (input.isAlive()) {
            @Nullable
            final Tensor delta = data.get(0);
            @Nonnull
            final Tensor feedback[] = new Tensor[itemCnt];
            com.simiacryptus.ref.wrappers.RefArrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
            delta.coordStream(false).forEach((inputCoord) -> {
              for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
                feedback[inputItem].add(inputCoord, delta.get(inputCoord));
              }
            });
            @Nonnull
            TensorArray tensorArray = new TensorArray(feedback);
            input.accumulate(buffer, tensorArray);
          }
        }) {

      @Override
      public boolean isAlive() {
        return input.isAlive();
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      @Nonnull DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    if (null != lastResult) {
      json.add("lastResult", lastResult.getJson(resources, dataSerializer));
    }
    json.addProperty("minBatches", minBatches);
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public void _free() {
    super._free();
  }

  public @Override @SuppressWarnings("unused") SumMetaLayer addRef() {
    return (SumMetaLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") SumMetaLayer[] addRefs(SumMetaLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SumMetaLayer::addRef)
        .toArray((x) -> new SumMetaLayer[x]);
  }

  public static @SuppressWarnings("unused") SumMetaLayer[][] addRefs(SumMetaLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SumMetaLayer::addRefs)
        .toArray((x) -> new SumMetaLayer[x][]);
  }
}
