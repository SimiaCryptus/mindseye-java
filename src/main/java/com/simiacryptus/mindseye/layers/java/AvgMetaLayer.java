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

@SuppressWarnings("serial")
public class AvgMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgMetaLayer.class);
  @Nullable
  public Tensor lastResult;
  private int minBatchCount = 1;

  public AvgMetaLayer() {
  }

  protected AvgMetaLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    lastResult = Tensor.fromJson(json.get("lastResult"), resources);
    minBatchCount = json.get("minBatchCount").getAsInt();
  }

  public int getMinBatchCount() {
    return minBatchCount;
  }

  @Nonnull
  public AvgMetaLayer setMinBatchCount(final int minBatchCount) {
    this.minBatchCount = minBatchCount;
    return this;
  }

  @SuppressWarnings("unused")
  public static AvgMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgMetaLayer(json, rs);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result input = inObj[0];
    TensorList inputData = input.getData();
    final int itemCnt = inputData.length();
    @Nullable
    Tensor thisResult;
    boolean passback;
    if (null == lastResult || inputData.length() > minBatchCount) {
      @Nonnull final ToDoubleFunction<Coordinate> f = (c) -> IntStream.range(0, itemCnt).mapToDouble(dataIndex -> {
        Tensor tensor = inputData.get(dataIndex);
        return tensor.get(c);
      }).sum() / itemCnt;
      Tensor tensor = inputData.get(0);
      thisResult = tensor.mapCoords(f);
      passback = true;
      lastResult = thisResult;
    } else {
      passback = false;
      thisResult = lastResult;
    }
    return new Result(new TensorArray(thisResult),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
          if (passback && input.isAlive()) {
            @Nullable final Tensor delta = data.get(0);
            @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
            Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
            thisResult.coordStream(true).forEach((inputCoord) -> {
              for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
                feedback[inputItem].add(inputCoord, delta.get(inputCoord) / itemCnt);
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

      @Override
      protected void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    if (null != lastResult) {
      json.add("lastResult", lastResult.getJson(resources, dataSerializer));
    }
    json.addProperty("minBatchCount", minBatchCount);
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

  @Override
  protected void _free() {
    super._free();
  }
}
